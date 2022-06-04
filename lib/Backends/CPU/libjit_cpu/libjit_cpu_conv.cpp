/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <cblas.h>

#include "../../../LLVMIRCodeGen/libjit/libjit_defs.h"

namespace {

// Initialize the convolution output frame for slice \p N with the bias \p
// biasW.
void libjit_conv_init_output_with_bias(dim_t N, float *outW, const float *biasW,
                                       const dim_t *outWdims,
                                       const dim_t *biasWdims) {
  // For each (x,y) step in the output tensor:
  for (dim_t ax = 0; ax < outWdims[1]; ax++) {
    for (dim_t ay = 0; ay < outWdims[2]; ay++) {
      // For each output channel:
      for (dim_t d = 0; d < outWdims[3]; d++) {
        // Store the results to the output buffer.
        float bias = biasW[d];
        auto outIdx = libjit_getXYZW(outWdims, N, ax, ay, d);
        outW[outIdx] = bias;
      } // For each depth in the output.
    }   // For each Y in the output.
  }     // For each X in the output.
}

/// Perform the heart of the convolution. Load \p ywidth scalars in a specific
/// channel, broadcast them, and multiply them with
/// [ywidth * float8 * numDepthRegs] depth values and accumulate them to create
/// [ywidth * float8 * numDepthRegs] depth result values.
void libjit_convDKKC8_convolve_channel(
    float *outW, const float *inW, const float *filterW, const dim_t *outWdims,
    const dim_t *inWdims, const dim_t *filterWdims, size_t sampleN,
    dim_t outChannel, unsigned numDepthRegs, unsigned ywidth, dim_t numChannels,
    sdim_t inX, sdim_t inY, sdim_t outX, sdim_t outY, size_t filterX,
    size_t filterY, size_t stride, size_t group) {

  // Process N * YWidth * 8 output pixels at once. Each value here is a
  // scalar that represents the sum for (x,y..y+ywidth) and the filter. The
  // SIMD dimension represents multiple layers of the depth
  // (output channel).
  float8 sum[numDepthRegs][ywidth];
  for (unsigned wu = 0; wu < ywidth; wu++) {
    for (unsigned du = 0; du < numDepthRegs; du++) {
      sum[du][wu] = BroadcastFloat8(0.0);
    }
  }

  // Perform the heart of the convolution.

  // For each input channel:
  for (size_t fd = 0; fd < numChannels; fd++) {
    // First, load and broadcast the scalar data from the input buffer.
    float8 in8[ywidth];
    for (unsigned wu = 0; wu < ywidth; wu++) {
      // Load a single pixel from the input image and broadcast it.
      auto inIdx = libjit_getXYZW(inWdims, sampleN, inX, inY + wu * stride,
                                  fd + group * numChannels);
      in8[wu] = BroadcastFloat8(inW[inIdx]);
    }

    // For each y pixel:
    for (unsigned wu = 0; wu < ywidth; wu++) {
      // Load N x 8 elements from the filter layer. The filter is
      // pre-swizzled to ensure efficient access.
      for (unsigned du = 0; du < numDepthRegs; du++) {
        auto filterIdx = libjit_getXYZWQ(filterWdims, outChannel / 8 + du,
                                         filterX, filterY, fd, 0);
        float8 ff0 = LoadFloat8(&filterW[filterIdx]);
        sum[du][wu] += ff0 * in8[wu];
      }
    }
  }

  // Store the results to the output buffer.
  for (unsigned wu = 0; wu < ywidth; wu++) {
    for (unsigned du = 0; du < numDepthRegs; du++) {
      // Add the partial sum to the tile.
      auto outIdx = libjit_getXYZW(outWdims, sampleN, outX, outY + wu,
                                   outChannel + du * 8);
      AddFloat8(&outW[outIdx], sum[du][wu]);
    }
  }
}

/// Process the input buffer in the convolution by iterating on the filter and
/// then on the pixels. This means that we process the whole input image for
/// each pixel in the filter. We try to unroll and process multiple inputs on
/// the Y row together.
void libjit_convDKKC8_foreach_xy_filter_pixels(
    size_t sampleN, dim_t outChannel, unsigned numDepthRegs,
    unsigned depthStrips, unsigned sizeGroupY, dim_t numChannels, float *outW,
    const float *inW, const float *filterW, const float *biasW,
    const dim_t *outWdims, const dim_t *inWdims, const dim_t *filterWdims,
    const dim_t *biasWdims, const dim_t *kernelSizes, const dim_t *strides,
    const dim_t *pads, dim_t group, dim_t endChannelIndex) {
  // The loops below look scary but the idea is simple. We iterate over
  // the pixels in the output tensor and calculate the coordinate of the source
  // tensor. When we process the Y row we try to process [sizeGroupY] elements
  // at once. After we finish the row we handle the odd cases by handling one y
  // value at a time.

  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  dim_t stride_w = strides[1];
  dim_t kernel_h = kernelSizes[0];
  dim_t kernel_w = kernelSizes[1];
  // For each element in the convolution-filter:
  for (dim_t fx = 0; fx < kernel_h; fx++) {
    for (dim_t fy = 0; fy < kernel_w; fy++) {

      // For each x step in the input/output tensor:
      for (dim_t outx = 0; outx < outWdims[1]; outx++) {
        sdim_t inx = (sdim_t)outx * stride_h - pad_t + fx;

        // Ignore out-of-bounds X values.
        if (inx < 0 || inx >= (sdim_t)inWdims[1]) {
          continue;
        }

        // For each y step in the input/output tensor, in steps of \p
        // sizeGroupY. We process \p sizeGroupY pixels of Y in one iteration.
        dim_t outy = 0;
        while (outy < outWdims[2]) {
          sdim_t iny = (sdim_t)outy * stride_w - pad_l + fy;

          if ((sdim_t)(iny + (sdim_t)stride_w * sizeGroupY) >=
              (sdim_t)inWdims[2]) {
            // If we've passed the upper bound, we don't want to increment
            // `outy` again, since we're going to handle the remaining y steps
            // in the following loop.
            break;
          }
          // Ignore out of bound indices.
          if (iny < 0) {
            /// We know iny is out of bounds, so we have nothing to do for outy.
            /// But we can't skip ahead by sizeGroupY, because we haven't
            /// checked outy + 1.
            outy += 1;
            continue;
          }

          // Convolve the (x,y .. y + ywidth) values.
          dim_t outC = outChannel;
          for (unsigned strip = 0;
               strip < depthStrips && outC < endChannelIndex; strip++) {
            libjit_convDKKC8_convolve_channel(
                outW, inW, filterW, outWdims, inWdims, filterWdims, sampleN,
                outC, numDepthRegs, sizeGroupY, numChannels, inx, iny, outx,
                outy, fx, fy, stride_w, group);
            outC += numDepthRegs * 8;
          }

          outy += sizeGroupY;
        } // For each Y group in the output.

        // Handle the remaining Y in the row in groups of size 1.
        for (; outy < outWdims[2]; outy++) {
          sdim_t iny = (sdim_t)outy * stride_w - pad_l + fy;
          // Ignore out of bound indices.
          if (iny < 0 || iny >= (sdim_t)inWdims[2]) {
            continue;
          }

          // Convolve a single (x,y) value.
          dim_t outC = outChannel;
          for (unsigned strip = 0;
               strip < depthStrips && outC < endChannelIndex; strip++) {
            libjit_convDKKC8_convolve_channel(
                outW, inW, filterW, outWdims, inWdims, filterWdims, sampleN,
                outC, numDepthRegs, 1, numChannels, inx, iny, outx, outy, fx,
                fy, stride_w, group);
            outC += numDepthRegs * 8;
          }
        } // For each Y, in step of 1, in the output.

      } // For each X in the output.
    }   // For each Y in the filter.
  }     // For each X in the filter.
}

// Process the input buffer in the convolution by iterating on the input buffer
// and then on the filter. This means that we process the whole input filter for
// each pixel in the input buffer.
void libjit_convDKKC8_foreach_xy_pixels_filter(
    size_t sampleN, dim_t outChannel, unsigned numDepthRegs,
    unsigned depthStrips, unsigned sizeGroupY, dim_t numChannels, float *outW,
    const float *inW, const float *filterW, const float *biasW,
    const dim_t *outWdims, const dim_t *inWdims, const dim_t *filterWdims,
    const dim_t *biasWdims, const dim_t *kernelSizes, const dim_t *strides,
    const dim_t *pads, dim_t group, dim_t endChannelIndex) {

  dim_t pad_t = pads[0];
  dim_t pad_l = pads[1];
  dim_t stride_h = strides[0];
  dim_t stride_w = strides[1];
  dim_t kernel_h = kernelSizes[0];
  dim_t kernel_w = kernelSizes[1];
  // For each (x,y) step in the input/output tensor:
  for (dim_t outx = 0; outx < outWdims[1]; outx++) {
    for (dim_t outy = 0; outy < outWdims[2]; outy++) {

      // For each element in the convolution-filter:
      for (dim_t fx = 0; fx < kernel_h; fx++) {
        for (dim_t fy = 0; fy < kernel_w; fy++) {

          // Calculate the specific input x,y that we process in this
          // iteration.
          dim_t inx = (dim_t)outx * stride_h - pad_t + fx;
          dim_t iny = (dim_t)outy * stride_w - pad_l + fy;

          // Ignore index access below zero (this is due to padding).
          if (inx < 0 || iny < 0 || inx >= inWdims[1] || iny >= inWdims[2]) {
            continue;
          }

          dim_t outC = outChannel;
          for (unsigned strip = 0;
               strip < depthStrips && outC < endChannelIndex; strip++) {
            libjit_convDKKC8_convolve_channel(
                outW, inW, filterW, outWdims, inWdims, filterWdims, sampleN,
                outC, numDepthRegs, 1, numChannels, inx, iny, outx, outy, fx,
                fy, stride_w, group);
            outC += numDepthRegs * 8;
          }
        } // For each Y in the filter.
      }   // For each X in the filter.
    }     // For each Y in the output.
  }       // For each X in the output.
}

} // namespace

extern "C" {
void libjit_convDKKC8_f(float *outW, const float *inW, const float *filterW,
                        const float *biasW, const dim_t *outWdims,
                        const dim_t *inWdims, const dim_t *filterWdims,
                        const dim_t *biasWdims, const dim_t *kernelSizes,
                        const dim_t *strides, const dim_t *pads, dim_t group,
                        unsigned pixelScanFirst, unsigned numDepthRegs,
                        unsigned sizeGroupY, unsigned depthStrips) {
  dim_t inChannels = inWdims[3];
  dim_t outChannels = outWdims[3];
  dim_t inCperG = inChannels / group;
  dim_t outCperG = outChannels / group;

  // Select the order in which we iterate over the pixels in the picture.
  auto eachPixelConv =
      (pixelScanFirst ? &libjit_convDKKC8_foreach_xy_pixels_filter
                      : &libjit_convDKKC8_foreach_xy_filter_pixels);

  // For each input in the batch:
  for (dim_t n = 0; n < inWdims[0]; n++) {

    // Initialize the output frame for the N'th slice with the bias.
    // Later we will accumulate values into this slice.
    libjit_conv_init_output_with_bias(n, outW, biasW, outWdims, biasWdims);

    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {

      // For each output channel, process [numDepthRegs x float8] elements.
      dim_t startChannelIndex = g * outCperG;
      dim_t endChannelIndex = (g + 1) * outCperG;
      for (dim_t d = startChannelIndex; d < endChannelIndex;
           d += 8 * numDepthRegs * depthStrips) {

        // Perform the convolution for each pixel.
        eachPixelConv(n, d, numDepthRegs, depthStrips, sizeGroupY, inCperG,
                      outW, inW, filterW, biasW, outWdims, inWdims, filterWdims,
                      biasWdims, kernelSizes, strides, pads, g,
                      endChannelIndex);

      } // For each D (the depth, or the output channel).
    }   // for each G, the group
  }     // For each N, the sample in the batch.
}

void libjit_naiveConvMO436_f(float *outW, const float *inW, const float *filterW,
                             const float *biasW, const dim_t *outWdims,
                             const dim_t *inWdims, const dim_t *filterWdims,
                             const dim_t *biasWdims, const dim_t *strides, const dim_t *pads)
  {
    dim_t numberOfFilters = filterWdims[0];
    dim_t filterHeight = filterWdims[1];
    dim_t filterWidth = filterWdims[2];

    dim_t numberOfInputs = inWdims[0];
    dim_t inputChannels = inWdims[3];

    dim_t outputHeight = outWdims[1];
    dim_t outputWidth = outWdims[2];

    dim_t stride = strides[0];
    dim_t padTop = pads[0];
    dim_t padLeft = pads[1];

    for (int inputNum = 0; inputNum < numberOfInputs; inputNum++)
    {
      libjit_conv_init_output_with_bias(inputNum, outW, biasW, outWdims, biasWdims);

      for (size_t filterNum = 0; filterNum < numberOfFilters; filterNum++)
      {
        for (size_t outI = 0; outI < outputHeight; outI++)
        {
          for (size_t outJ = 0; outJ < outputWidth; outJ++)
          {
            float temp = .0;

            for (size_t inputCh = 0; inputCh < inputChannels; inputCh++)
            {
              for (size_t filterY = 0; filterY < filterHeight; filterY++)
              {
                for (size_t filterX = 0; filterX < filterWidth; filterX++)
                {
                  auto inX = outI * stride - padTop + filterX;
                  auto inY = outJ * stride - padLeft + filterY;

                  // ignore if indexes are out of boundaries
                  if (inX < 0 || inY < 0 || inX >= (sdim_t)inWdims[1] || inY >= (sdim_t)inWdims[2])
                  {
                    continue;
                  }

                  auto inIdx = libjit_getXYZW(inWdims, inputNum, inX, inY, inputCh);

                  auto filterIdx = libjit_getXYZW(filterWdims, filterNum, filterX,
                                                  filterY, inputCh);

                  temp += inW[inIdx] * filterW[filterIdx];
                } // Filter X (width)
              }   // Filter Y (height)
            }     // Input channel

            // since the number of filters and output channels are same, i'm using filterNum because I already have that value in this context.
            auto outIdx = libjit_getXYZW(outWdims, inputNum, outI, outJ, filterNum);

            outW[outIdx] = outW[outIdx] + temp;
          } // Output Width
        }   // Output Height
      }     // Filter #
  }       // Input #
}

/**
 * This method was modified/adapted from Caffe2 Im2ColNHWC source code.
 * https://github.com/pytorch/pytorch/blob/7e4730d017e3da94fe3b7b5f504df0563516367f/caffe2/quantization/server/im2col_dnnlowp.h#L186
 * 
 * Copyright (c) 2016-present, Facebook Inc. All rights reserved.
 *
 * All contributions by Facebook:
 * Copyright (c) 2016 Facebook Inc.
 * 
 * All contributions by Google:
 * Copyright (c) 2015 Google Inc.
 * All rights reserved.
 * 
 * All contributions by Yangqing Jia:
 * Copyright (c) 2015 Yangqing Jia
 * All rights reserved.
 * 
 * All contributions by Kakao Brain:
 * Copyright 2019-2020 Kakao Brain
 * 
 * All contributions by Cruise LLC:
 * Copyright (c) 2022 Cruise LLC.
 * All rights reserved.
 * 
 * All contributions from Caffe:
 * Copyright(c) 2013, 2014, 2015, the respective contributors
 * All rights reserved.
 * 
 * All other contributions:
 * Copyright(c) 2015, 2016 the respective contributors
 * All rights reserved.
 * 
 * Caffe2 uses a copyright model similar to Caffe: each contributor holds
 * copyright over their contributions to Caffe2. The project versioning records
 * all such contribution and copyright details. If a contributor wants to further
 * mark their specific copyright on a particular contribution, they should
 * indicate their copyright solely in the commit message of the change when it is
 * committed.
 * 
 * All rights reserved.
 * 
 * LICENSE: https://raw.githubusercontent.com/pytorch/pytorch/7e4730d017e3da94fe3b7b5f504df0563516367f/LICENSE
 */
static void Im2ColNHWC(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* data_im,
    float* data_col,
    const int groups,
    const float& zero_point)
{
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  for (int h = 0; h < height_col; ++h) {
    int h_pad = -pad_t + h * stride_h;
    float* data_col_temp =
        data_col + h * width_col * kernel_h * kernel_w * channels;
    int w_pad = -pad_l;
    for (int w = 0 ; w < width_col; w++) {
      int r = 0;
      for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
        int s = 0;
        for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w, ++s) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            for (int g = 0 ; g < groups ; g++) {
              memcpy(
                  data_col_temp +
                      ((g * kernel_h + r) * kernel_w + s) * (channels / groups),
                  data_im + (ih * width + iw) * channels +
                      g * (channels / groups),
                  sizeof(float) * (channels / groups));
            }
          } else {
            // This should be simply padded with zero.
            for (int g = 0 ; g < groups; g++) {
              for (int i = 0; i < channels / groups; ++i) {
                data_col_temp
                    [(((g * kernel_h + r) * kernel_w) + s) *
                         (channels / groups) +
                     i] = zero_point;
              }
            }
          }
        } // for each iw
      } // for each ih
      data_col_temp += kernel_h * kernel_w * channels;
      w_pad += stride_w;
    } // for each output pixel
  } // for each image row
}


void libjit_gemmConvMO436_f(float *outW, const float *inW, const float *filterW,
                           const float *biasW, const dim_t *outWdims,
                           const dim_t *inWdims, const dim_t *filterWdims,
                           const dim_t *biasWdims, const dim_t *strides, const dim_t *pads)
{
    dim_t numberOfFilters = filterWdims[0];
    dim_t filterHeight = filterWdims[1];
    dim_t filterWidth = filterWdims[2];

    dim_t inputHeight = inWdims[1];
    dim_t inputWidth = inWdims[2];
    dim_t inputChannels = inWdims[3];

    dim_t outputHeight = outWdims[1];
    dim_t outputWidth = outWdims[2];

    dim_t strideHeight = strides[0];
    dim_t strideWidth = strides[1];

    dim_t padTop = pads[0];
    dim_t padLeft = pads[1];
    dim_t padBottom = pads[2];
    dim_t padRight = pads[3];

    dim_t dilationHeight = 1;
    dim_t dilationWidth = 1;

    dim_t groups = 1;

    int imColRows = outputHeight * outputWidth;
    int imColColumns = filterHeight * filterWidth * inputChannels;
    int size = imColRows * imColColumns;

    libjit_conv_init_output_with_bias(0, outW, biasW, outWdims, biasWdims);

    float *imgCol = (float *)malloc(size* sizeof(float));

    Im2ColNHWC(
        inputChannels,
        inputHeight,
        inputWidth,
        filterHeight,
        filterWidth,
        dilationHeight,
        dilationWidth,
        padTop,
        padLeft,
        padBottom,
        padRight,
        strideHeight,
        strideWidth,
        inW,
        imgCol,
        groups,
        0 // zero_point
    );

    float alpha = 1.0;
    float beta = 1.0;

    int M = imColRows;
    int N = numberOfFilters;
    int K = filterHeight * filterWidth * inputChannels;

    int lda = K;
    int ldb = K;
    int ldc = N;

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        M, N, K,
        alpha,
        imgCol, // A
        lda,
        filterW, // B
        ldb,
        beta,
        outW, ldc);

    free(imgCol);
}

} // extern "C"