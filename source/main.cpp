#include <algorithm>
#include <array>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <fstream>
#include <iostream>
#include <sstream>
#define FatalError(s)                                                          \
  do {                                                                         \
    {                                                                          \
      std::stringstream _message;                                              \
      (_message << __FILE__ << ':' << __LINE__ << ' ' << std::string(s)        \
                << std::endl);                                                 \
      (std::cerr << _message.str());                                           \
      cudaDeviceReset();                                                       \
    }                                                                          \
  } while (0)
#define checkCuda(status)                                                      \
  do {                                                                         \
    {                                                                          \
      std::stringstream _error;                                                \
      if ((0 != status)) {                                                     \
        (_error << " Cuda failure: " << cudaGetErrorString(status));           \
        FatalError(_error.str());                                              \
      }                                                                        \
    }                                                                          \
  } while (0)
#define checkCufft(status)                                                     \
  do {                                                                         \
    {                                                                          \
      std::stringstream _error;                                                \
      if ((CUFFT_SUCCESS != status)) {                                         \
        (_error << "CUFFT failure: " << cufft_get_error_string(status));       \
        FatalError(_error.str());                                              \
      }                                                                        \
    }                                                                          \
  } while (0)
// float data[NZ][NY][NX]
// float* flattened = data
// data[z][y][x] == flattened[x+NX*y+NX*NY*z]
// array<array<array<float,x>,y>,z> data_
// float* flattened = data_
// data_.at(x).at(y).at(z) == flattened[x+NX*y+NX*NY*z]
#define ft_idx(x, y, z) ((z) + (y * NZ) + (x * NZ * NY))
#define ft_idx2(x, y) ((x) + (y * NX))
enum ft_constants_t { NX = 256, NY = 256, NZ = 256 };

cufftComplex *ft_data;
cufftHandle ft_plan;

const char *cufft_get_error_string(cufftResult_t result) {
  {
    const std::array<const char *, 17> msg(
        {{"SUCCESS", "INVALID_PLAN", "ALLOC_FAILED", "INVALID_TYPE",
          "INVALID_VALUE", "INTERNAL_ERROR", "EXEC_FAILED", "SETUP_FAILED",
          "INVALID_SIZE", "UNALIGNED_DATA", "INCOMPLETE_PARAMETER_LIST",
          "INVALID_DEVICE", "PARSE_ERROR", "NO_WORKSPACE", "NOT_IMPLEMENTED",
          "LICENSE_ERROR", "NOT_SUPPORTED"}});
    return msg.at(result);
  }
}
void ft_init() {
  checkCuda(cudaMallocManaged(reinterpret_cast<void **>(&ft_data),
                              (sizeof(*ft_data) * NX * NY * NZ),
                              cudaMemAttachGlobal));
  checkCufft(cufftPlan3d(&ft_plan, NX, NY, NZ, CUFFT_C2C));
}
void ft_fill_sinc(cufftComplex *data, float radius = (1.e+0f)) {
  for (unsigned int k = 0; (k < NZ); k += 1) {
    for (unsigned int j = 0; (j < NY); j += 1) {
      for (unsigned int i = 0; (i < NX); i += 1) {
        {
          auto x(((-5.e-1f) + (static_cast<float>(i) / NX)));
          auto y(((-5.e-1f) + (static_cast<float>(j) / NY)));
          auto z(((-5.e-1f) + (static_cast<float>(k) / NZ)));
          auto r((2 * static_cast<float>(M_PI) * radius *
                  std::sqrt(((x * x) + (y * y) + (z * z)))));
          auto sign((-1 + (2 * ((i + j + k) % 2))));
          auto alpha((5.4e-1f));
          auto beta((1 - alpha));
          auto hamming_x((alpha - (beta * std::cos(static_cast<float>(
                                              ((2 * M_PI * i) / (NX - 1)))))));
          auto hamming_y((alpha - (beta * std::cos(static_cast<float>(
                                              ((2 * M_PI * j) / (NY - 1)))))));
          auto hamming_z((alpha - (beta * std::cos(static_cast<float>(
                                              ((2 * M_PI * k) / (NZ - 1)))))));
          auto hamming((hamming_x * hamming_y * hamming_z));
          if (((0.0e+0f) == r)) {
            data[ft_idx(i, j, k)] = {(sign * (1.e+0f)), (0.0e+0f)};
          } else {
            data[ft_idx(i, j, k)] = {((hamming * sign * std::sin(r)) / r),
                                     (0.0e+0f)};
          }
        }
      }
    }
  }
}
void ft_delete() {
  checkCufft(cufftDestroy(ft_plan));
  checkCuda(cudaFree(ft_data));
}
void pgm_write_xy(std::string fn, cufftComplex *data, size_t z0,
                  float scale = (0.0e+0f)) {
  {
    std::ofstream f(fn, (std::ofstream::out | std::ofstream::binary |
                         std::ofstream::trunc));
    unsigned char *bufu8(new unsigned char[(NX * NY)]);
    if (((0.0e+0f) == scale)) {
      {
        std::array<float, NX * NY> buff32;
        for (unsigned int j = 0; (j < NY); j += 1) {
          for (unsigned int i = 0; (i < NX); i += 1) {
            {
              auto sign((-1 + (2 * ((i + j + z0) % 2))));
              buff32[ft_idx2(i, j)] = (cuCabsf(ft_data[ft_idx(i, j, z0)]));
            }
          }
        }
        {
          auto minmax(std::minmax_element(buff32.begin(), buff32.end()));
          auto mi(minmax.first[0]);
          auto ma(minmax.second[0]);
          for (auto &e : buff32) {
            e = (((2.55e+2f) * (e - mi)) / (ma - mi));
          }
        }
        for (unsigned int i = 0; (i < buff32.size()); i += 1) {
          bufu8[i] = static_cast<int>(buff32[i]);
        }
      }
    } else {
      for (unsigned int j = 0; (j < NY); j += 1) {
        for (unsigned int i = 0; (i < NX); i += 1) {
          {
            auto sign((-1 + (2 * ((i + j + z0) % 2))));
            bufu8[ft_idx2(i, j)] = std::min(
                255, std::max(0, static_cast<int>(
                                     (scale * sign *
                                      cuCrealf(ft_data[ft_idx(i, j, z0)])))));
          }
        }
      }
    }
    (f << "P5\n" << NX << " " << NY << "\n255\n");
    f.write(reinterpret_cast<char *>(bufu8), (NX * NY));
  }
}
int main(int argc, char **argv) {
  ft_init();
  ft_fill_sinc(ft_data, static_cast<float>((6.e+1f)));
  checkCufft(cufftExecC2C(ft_plan, ft_data, ft_data, CUFFT_FORWARD));
  cudaDeviceSynchronize();
  pgm_write_xy("/dev/shm/o.pgm", ft_data, (NZ / 2));
  ft_delete();
}