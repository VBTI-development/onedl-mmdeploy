// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/segmentor.h"

#include "common.h"

namespace mmdeploy::python {

class PySegmentor {
 public:
  PySegmentor(const char* model_path, const char* device_name, int device_id) {
    auto status =
        mmdeploy_segmentor_create_by_path(model_path, device_name, device_id, &segmentor_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create segmentor");
    }
  }
  ~PySegmentor() {
    mmdeploy_segmentor_destroy(segmentor_);
    segmentor_ = {};
  }

  std::vector<py::array> Apply(const std::vector<PyImage>& imgs) {
    std::vector<mmdeploy_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto& img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mmdeploy_segmentation_t* segm{};
    auto status = mmdeploy_segmentor_apply(segmentor_, mats.data(), (int)mats.size(), &segm);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply segmentor, code: " + std::to_string(status));
    }
    using Sptr = std::shared_ptr<mmdeploy_segmentation_t>;
    Sptr holder(segm, [n = mats.size()](auto p) { mmdeploy_segmentor_release_result(p, n); });
    // Create a single capsule for all arrays to ensure correct buffer lifetime
    py::capsule cap(new Sptr(holder), [](void* p) { delete reinterpret_cast<Sptr*>(p); });

    std::vector<py::array> rets(mats.size());
    for (size_t i = 0; i < mats.size(); ++i) {
      if (segm[i].mask != nullptr) {
        if (segm[i].mask_dtype == 1) {  // int32
          rets[i] = py::array_t<int32_t>({segm[i].height, segm[i].width},
                                         static_cast<const int32_t*>(segm[i].mask), cap);
        } else if (segm[i].mask_dtype == 2) {  // int64
          rets[i] = py::array_t<int64_t>({segm[i].height, segm[i].width},
                                         static_cast<const int64_t*>(segm[i].mask), cap);
        } else {
          throw std::runtime_error("Unsupported mask dtype in Python binding");
        }
      }
      if (segm[i].score != nullptr) {
        rets[i] = py::array_t<float>({segm[i].classes, segm[i].height, segm[i].width},
                                     segm[i].score, cap);
      }
    }

    return rets;
  }

 private:
  mmdeploy_segmentor_t segmentor_{};
};

static PythonBindingRegisterer register_segmentor{[](py::module& m) {
  py::class_<PySegmentor>(m, "Segmentor")
      .def(py::init([](const char* model_path, const char* device_name, int device_id) {
             return std::make_unique<PySegmentor>(model_path, device_name, device_id);
           }),
           py::arg("model_path"), py::arg("device_name"), py::arg("device_id") = 0)
      .def("__call__",
           [](PySegmentor* self, const PyImage& img) -> py::array {
             return self->Apply(std::vector{img})[0];
           })
      .def("batch", &PySegmentor::Apply);
}};

}  // namespace mmdeploy::python
