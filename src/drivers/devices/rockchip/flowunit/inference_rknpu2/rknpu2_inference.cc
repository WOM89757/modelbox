/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rknpu2_inference.h"

#include <model_decrypt.h>
#include <modelbox/base/log.h>
#include <modelbox/base/status.h>

#include <algorithm>

#include "modelbox/device/rockchip/rockchip_memory.h"
#include "securec.h"

#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

static std::map<std::string, rknn_tensor_type> type_map = {
    {"FLOAT", RKNN_TENSOR_FLOAT32},   {"INT", RKNN_TENSOR_INT32},
    {"FLOAT32", RKNN_TENSOR_FLOAT32}, {"FLOAT16", RKNN_TENSOR_FLOAT16},
    {"INT8", RKNN_TENSOR_INT8},       {"UINT8", RKNN_TENSOR_UINT8},
    {"INT16", RKNN_TENSOR_INT16},     {"UINT16", RKNN_TENSOR_UINT16},
    {"INT32", RKNN_TENSOR_INT32},     {"UINT32", RKNN_TENSOR_UINT32},
    {"INT64", RKNN_TENSOR_INT64}};

static std::map<rknn_tensor_type, size_t> type_size_map = {
    {RKNN_TENSOR_FLOAT32, 4}, {RKNN_TENSOR_FLOAT16, 2}, {RKNN_TENSOR_INT8, 1},
    {RKNN_TENSOR_UINT8, 1},   {RKNN_TENSOR_INT16, 2},   {RKNN_TENSOR_UINT16, 2},
    {RKNN_TENSOR_INT32, 4},   {RKNN_TENSOR_UINT32, 4},  {RKNN_TENSOR_INT64, 8}};

modelbox::Status modelbox::RKNPU2Inference::LoadModel(
    const std::string &model_file,
    const std::shared_ptr<modelbox::Drivers> &drivers_ptr,
    const std::shared_ptr<modelbox::Configuration> &config) {
  ModelDecryption rknpu2_model_decrypt;
  if (modelbox::STATUS_SUCCESS !=
      rknpu2_model_decrypt.Init(model_file, drivers_ptr, config)) {
    MBLOG_ERROR << "init model fail";
    return modelbox::STATUS_FAULT;
  }

  int64_t model_len = 0;
  std::shared_ptr<uint8_t> modelBuf =
      rknpu2_model_decrypt.GetModelSharedBuffer(model_len);
  if (!modelBuf) {
    MBLOG_ERROR << "GetDecryptModelBuffer fail";
    return modelbox::STATUS_FAULT;
  }

  int ret = rknn_init(&ctx_, modelBuf.get(), model_len, 0, nullptr);
  if (ret != RKNN_SUCC) {
    MBLOG_ERROR << "rknn_init fail:" << ret;
    ctx_ = 0;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status modelbox::RKNPU2Inference::ConvertType(
    const std::string &type, rknn_tensor_type &rk_type) {
  auto tmp_type = type;
  std::transform(tmp_type.begin(), tmp_type.end(), tmp_type.begin(), ::toupper);
  auto iter = type_map.find(tmp_type);
  if (iter == type_map.end()) {
    MBLOG_ERROR << "Not support type: " << type;
    return modelbox::STATUS_FAULT;
  }
  rk_type = iter->second;
  return modelbox::STATUS_OK;
}

modelbox::Status modelbox::RKNPU2Inference::GetModelAttr() {
  inputs_type_.resize(npu2model_input_list_.size());
  inputs_size_.resize(npu2model_input_list_.size());
  // rknn_tensor_attr use new to avoid stack crash
  std::shared_ptr<rknn_tensor_attr> tmp_attr =
      std::make_shared<rknn_tensor_attr>();
  for (size_t i = 0; i < npu2model_input_list_.size(); i++) {
    tmp_attr->index = (unsigned int)i;
    auto ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, tmp_attr.get(),
                          sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      MBLOG_ERROR << "query input attrs error";
      return {modelbox::STATUS_FAULT, "query input attrs error"};
    }

    rknn_tensor_type rk_type;
    auto status = ConvertType(npu2model_type_list_[i], rk_type);
    if (status != modelbox::STATUS_OK) {
      MBLOG_ERROR << "input type convert failed. " << status.WrapErrormsgs();
      return {status, "input type convert failed."};
    }
    inputs_type_[i] = rk_type;
    inputs_size_[i] = tmp_attr->n_elems * type_size_map[rk_type];
    MBLOG_INFO << "------tmp_attr nelems " << tmp_attr->n_elems << " " << type_size_map[rk_type]  << " " << inputs_size_[i];
  }

  outputs_size_.resize(npu2model_output_list_.size());
  for (size_t i = 0; i < npu2model_output_list_.size(); i++) {
    tmp_attr->index = (unsigned int)i;
    auto ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, tmp_attr.get(),
                          sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      MBLOG_ERROR << "query output attrs error";
      return {modelbox::STATUS_FAULT, "query output attrs error"};
    }

    rknn_tensor_type rk_type;
    auto status = ConvertType(npu2model_type_list_output_[i], rk_type);
    if (status != modelbox::STATUS_OK) {
      MBLOG_ERROR << "output type convert failed. " << status.WrapErrormsgs();
      return {status, "output type convert failed."};
    }
    outputs_size_[i] = tmp_attr->n_elems * type_size_map[rk_type];
    MBLOG_INFO << "------ouput tmp_attr nelems " << tmp_attr->n_elems << " " << type_size_map[rk_type]  << " " << outputs_size_[i];
  }
  return STATUS_SUCCESS;
}

modelbox::Status modelbox::RKNPU2Inference::Init(
    const std::string &model_file,
    const std::shared_ptr<modelbox::Drivers> &drivers_ptr,
    const std::shared_ptr<modelbox::Configuration> &config,
    const std::shared_ptr<modelbox::InferenceRKNPUParams> &params) {
  batch_size_ = config->GetInt32("batch_size", 1);

  if (LoadModel(model_file, drivers_ptr, config) != STATUS_SUCCESS) {
    return modelbox::STATUS_FAULT;
  }
  // just use input name without check
  npu2model_input_list_ = params->input_name_list_;
  npu2model_type_list_ = params->input_type_list_;
  npu2model_output_list_ = params->output_name_list_;
  npu2model_type_list_output_ = params->output_type_list_;

  rknn_input_output_num rknpu2_io_num;
  auto ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &rknpu2_io_num,
                        sizeof(rknpu2_io_num));
  if (ret != RKNN_SUCC) {
    MBLOG_ERROR << "query input_output error";
    return {modelbox::STATUS_FAULT, "query input_output error"};
  }

  if (npu2model_input_list_.size() != rknpu2_io_num.n_input ||
      npu2model_output_list_.size() != rknpu2_io_num.n_output) {
    MBLOG_ERROR << "model input output num mismatch: input num in graph is "
                << npu2model_input_list_.size()
                << ", the real model input num is " << rknpu2_io_num.n_input
                << ", output num in graph is " << npu2model_output_list_.size()
                << "the real model output num is " << rknpu2_io_num.n_output;
    return modelbox::STATUS_FAULT;
  }
  auto device_type = RKNNDevs::Instance().GetDeviceType();
  if (device_type == 3) {
    auto core_id = config->GetUint64("core_mask_id", 0);
    MBLOG_INFO << "------------device is 3588, set core mask is " << core_id;
    auto ret = rknn_set_core_mask(ctx_, (rknn_core_mask)core_id);
    if (ret != RKNN_SUCC) {
      MBLOG_ERROR << "set core mask error";
      return {modelbox::STATUS_FAULT, "set core mask error"};
    }
  }
  

  return GetModelAttr();
}

modelbox::Status modelbox::RKNPU2Inference::Build_Outputs(
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  auto out_cnt = npu2model_output_list_.size();
  std::vector<rknn_output> rknpu2_outputs;
  rknpu2_outputs.reserve(out_cnt);

  for (size_t i = 0; i < out_cnt; ++i) {
    auto &name = npu2model_output_list_[i];
    auto buffer_list = data_ctx->Output(name);

    std::vector<size_t> shape({outputs_size_[i]});
    buffer_list->Build(shape, false);
    auto rknpu2_buffer = buffer_list->At(0);
    auto *mpp_buf = (MppBuffer)(rknpu2_buffer->MutableData());
    if (mpp_buf == nullptr) {
      MBLOG_INFO << "-------------mpp_buf is nullptr";
    }
    
    auto *data_buf = (float *)mpp_buffer_get_ptr(mpp_buf);

    // convert outputs to float*
    rknpu2_outputs.push_back({.want_float = true,
                              .is_prealloc = true,
                              .index = (unsigned int)i,
                              .buf = data_buf,
                              .size = (uint32_t)outputs_size_[i]});
    rknpu2_buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_FLOAT);
    rknpu2_buffer->Set("shape", outputs_size_[i]);
  }

  auto ret = rknn_outputs_get(ctx_, out_cnt, rknpu2_outputs.data(), nullptr);
  // reset rknpu2_outputs, avoid buf released
  for (auto ele : rknpu2_outputs) {
    ele.is_prealloc = 1;
    ele.buf = nullptr;
  }
  rknn_outputs_release(ctx_, out_cnt, rknpu2_outputs.data());
  if (ret != RKNN_SUCC) {
    MBLOG_ERROR << "rknn get output error";
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status modelbox::RKNPU2Inference::Build_Batch_Outputs(
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  auto out_cnt = npu2model_output_list_.size();
  std::vector<rknn_output> rknpu2_outputs;
  rknpu2_outputs.reserve(out_cnt);

  for (size_t i = 0; i < out_cnt; ++i) {
    uint32_t batch_iter_size = outputs_size_[i]/ this->batch_size_;
    // MBLOG_INFO << "##data output name: " <<  name  << " output_size: " << outputs_size_[i];
    // ##data output name: output output_size: 8294400
    // ##data output name: output2 output_size: 2073600
    // ##data output name: output3 output_size: 518400

    rknpu2_outputs.push_back({.want_float = true,
                          .is_prealloc = false,
                          .index = (unsigned int)i,
                          .size = batch_iter_size});
  }
  
  auto ret = rknn_outputs_get(ctx_, out_cnt, rknpu2_outputs.data(), nullptr);
  if (ret != RKNN_SUCC) {
    MBLOG_ERROR << "rknn get output error";
    return modelbox::STATUS_FAULT;
  }

  for (size_t i = 0; i < out_cnt; ++i) {
    auto &name = npu2model_output_list_[i];
    auto buffer_list = data_ctx->Output(name);

    uint32_t batch_iter_size = outputs_size_[i]/ this->batch_size_;
    uint32_t iter_size = batch_iter_size/sizeof(float);

    // std::vector<size_t> shape({outputs_size_[i]});
    std::vector<size_t> shape(real_batch, batch_iter_size);
    buffer_list->Build(shape, false);
    // MBLOG_INFO << "## buffer list size " << buffer_list.get()->Size();
    float *buf_loc = (float*)rknpu2_outputs[i].buf;
    for (size_t j = 0; j < buffer_list->Size(); j++) {
      
      auto rknpu2_buffer = buffer_list->At(j);
      // MBLOG_INFO << "##buffer size: " << rknpu2_buffer->GetBytes();
      auto *mpp_buf = (MppBuffer)(rknpu2_buffer->MutableData());
      if (mpp_buf == nullptr) {
        // MBLOG_INFO << "-------------mpp_buf is nullptr";
      }
      
      // MBLOG_INFO << "-------------mpp_buf is " << mpp_buf;
      auto *data_buf = (float *)mpp_buffer_get_ptr(mpp_buf);
      // MBLOG_INFO << "-------------data_buf is " << data_buf;

      // copy outputs to float*
      memcpy(data_buf, buf_loc, batch_iter_size);
      buf_loc += iter_size;

      // MBLOG_INFO << "-------------output size " << rknpu2_outputs[i].size;
      rknpu2_buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_FLOAT);
      rknpu2_buffer->Set("shape", batch_iter_size);
    }


  }
  rknn_outputs_release(ctx_, out_cnt, rknpu2_outputs.data());
  return modelbox::STATUS_SUCCESS;
}

size_t modelbox::RKNPU2Inference::CopyFromAlignMemory(
    std::shared_ptr<modelbox::BufferList> &input_buf_list,
    std::shared_ptr<uint8_t> &pdst,
    std::shared_ptr<modelbox::InferenceInputParams> &input_params) {
  int32_t one_size = input_params->in_width_ * input_params->in_height_;
  RgaSURF_FORMAT rga_fmt = RK_FORMAT_UNKNOWN;
  rga_fmt = modelbox::GetRGAFormat(input_params->pix_fmt_);

  size_t input_total_size = real_batch * one_size;
  if (rga_fmt == RK_FORMAT_YCbCr_420_SP || rga_fmt == RK_FORMAT_YCrCb_420_SP) {
    input_total_size = input_total_size * 3 / 2;
  } else {
    if (rga_fmt == RK_FORMAT_RGB_888 || rga_fmt == RK_FORMAT_BGR_888) {
      input_total_size = input_total_size * 3;
    }
  }

  if ((real_batch == 1) &&
      ((input_params->in_width_ == input_params->in_wstride_ &&
        input_params->in_height_ == input_params->in_hstride_) ||
       (input_params->in_wstride_ == 0 && input_params->in_hstride_ == 0))) {
    auto in_image = input_buf_list->At(0);
    auto *mpp_buf = (MppBuffer)(in_image->ConstData());
    auto *cpu_buf = (uint8_t *)mpp_buffer_get_ptr(mpp_buf);
    pdst.reset(cpu_buf, [](uint8_t *p) {});
    return input_total_size;
  }

  pdst.reset(new u_int8_t[input_total_size],
             [](const uint8_t *p) { delete[] p; });

  uint8_t *pdst_buf = pdst.get();
  for (size_t i = 0; i < real_batch; i++) {
    auto in_image = input_buf_list->At(i);
    auto *mpp_buf = (MppBuffer)(in_image->ConstData());
    auto *cpu_buf = (uint8_t *)mpp_buffer_get_ptr(mpp_buf);

    if (rga_fmt == RK_FORMAT_YCbCr_420_SP ||
        rga_fmt == RK_FORMAT_YCrCb_420_SP) {
      modelbox::CopyNVMemory(
          cpu_buf, pdst_buf, input_params->in_width_, input_params->in_height_,
          input_params->in_wstride_, input_params->in_hstride_);
      pdst_buf += one_size * 3 / 2;
    } else if (rga_fmt == RK_FORMAT_RGB_888 || rga_fmt == RK_FORMAT_BGR_888) {
      modelbox::CopyRGBMemory(
          cpu_buf, pdst_buf, input_params->in_width_, input_params->in_height_,
          input_params->in_wstride_, input_params->in_hstride_);
      pdst_buf += one_size * 3;
    } else {
      auto rc = memcpy_s(pdst_buf, one_size, cpu_buf, one_size);
      if (rc != EOK) {
        MBLOG_WARN << "RKNPUInference2 copy fail";
      }
      pdst_buf += one_size;
    }
  }

  return input_total_size;
}

size_t modelbox::RKNPU2Inference::GetInputBuffer(
    std::shared_ptr<uint8_t> &input_buf,
    std::shared_ptr<modelbox::BufferList> &input_buf_list) {
  auto in_image = input_buf_list->At(0);
  // MBLOG_INFO << "-------in image type " << (int)in_image->GetBufferType() << std::endl;
  // cv::Mat infer_img = cv::Mat(cv::Size(416,416), CV_8UC3, in_image.get()); 
  // cv::imwrite("infer-pre.jpg", infer_img);
  // exit(0);
  auto input_params = std::make_shared<InferenceInputParams>();
  input_params->pix_fmt_ = "";

  in_image->Get("width", input_params->in_width_);
  in_image->Get("height", input_params->in_height_);
  in_image->Get("width_stride", input_params->in_wstride_);
  in_image->Get("height_stride", input_params->in_hstride_);
  in_image->Get("pix_fmt", input_params->pix_fmt_);
  if (input_params->pix_fmt_ == "rgb" || input_params->pix_fmt_ == "bgr") {
    input_params->in_wstride_ /= 3;
  } else if (input_params->pix_fmt_.empty()) {
    input_params->in_height_ = 1;
    input_params->in_width_ = in_image->GetBytes();
  }

  // MBLOG_INFO << "---------get input buffer by " << input_buf_list->GetDevice()->GetType();
  // MBLOG_INFO << "-----" << in_image.get() ;
  if (input_buf_list->GetDevice()->GetType() == "rknpu" || input_buf_list->GetDevice()->GetType() == "rockchip") {
    // MBLOG_INFO << "input bufer list type is " << input_buf_list->GetDevice()->GetType();
    return CopyFromAlignMemory(input_buf_list, input_buf, input_params);
  }
  input_buf.reset((uint8_t *)input_buf_list->ConstData(), [](uint8_t *p) {});
  return input_buf_list->GetBytes();
}

int64_t getCurrentTime()      
    {    
       struct timeval tv;    
       gettimeofday(&tv, NULL);    
       return tv.tv_sec * 1000 + tv.tv_usec / 1000;    
    }  
double timestamp_now_float() {
        // return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
        return getCurrentTime();
    }
modelbox::Status modelbox::RKNPU2Inference::Infer(
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  // 构造impl的输入
  if (ctx_ == 0) {
    MBLOG_ERROR << "rk model not load, pass";
    return {STATUS_FAULT, "rk model not load, pass"};
  }

  std::vector<rknn_input> rknpu2_inputs;
  rknpu2_inputs.reserve(npu2model_input_list_.size());
  std::vector<std::shared_ptr<uint8_t>> rknpu2_input_bufs;
  rknpu2_input_bufs.resize(npu2model_input_list_.size());

  // MBLOG_INFO << "-----npu2model input list size " << npu2model_input_list_.size();
  for (size_t i = 0; i < npu2model_input_list_.size(); i++) {
    auto inputs = data_ctx->Input(npu2model_input_list_[i]);
    rknn_input one_input;
    real_batch = inputs->Size();
    if (real_batch != batch_size_) {
      auto msg = npu2model_input_list_[i] +
                 " batch mismatch:" + std::to_string(batch_size_) + " " +
                 std::to_string(real_batch);
      MBLOG_DEBUG << msg;
      // return {STATUS_FAULT, msg};
    }

    size_t ret_size = GetInputBuffer(rknpu2_input_bufs[i], inputs);
    one_input.index = i;
    one_input.buf = rknpu2_input_bufs[i].get();
    one_input.size = ret_size;
    // std::cout << "========== input " << inputs_type_[i] << std::endl;
    if (one_input.size != inputs_size_[i]) {
      MBLOG_DEBUG << "input size mismatch:(yours model) " << one_input.size
                  << " " << inputs_size_[i];
      // return modelbox::STATUS_FAULT;
    }
    one_input.pass_through = false;
    // inputs_type_[i] = 3;
    one_input.type = (rknn_tensor_type)inputs_type_[i];
    one_input.fmt = RKNN_TENSOR_NHWC;
    rknpu2_inputs.push_back(one_input);
  }

  std::lock_guard<std::mutex> lk(rknpu2_infer_mtx_);
  // MBLOG_INFO << "----------start rknn inputs set ";
  // MBLOG_INFO << "---- rknpu2 size " << rknpu2_inputs.size() << " data: " << rknpu2_inputs.data();
  
  
  // auto begin_timer = timestamp_now_float();
  auto ret = rknn_inputs_set(ctx_, rknpu2_inputs.size(), rknpu2_inputs.data());

  // std::cout << "----------end rknn inputs set " << std::endl;
  if (ret != RKNN_SUCC) {
    MBLOG_ERROR << "rknn_inputs_set fail: " << ret;
    return modelbox::STATUS_FAULT;
  }

  ret = rknn_run(ctx_, nullptr);
  if (ret != RKNN_SUCC) {
    MBLOG_ERROR << "run error fail: " << ret;
    return modelbox::STATUS_FAULT;
  }
  // MBLOG_INFO << "---------finished rknn run-----";
  // float inference_time = (timestamp_now_float() - begin_timer);
  // float inference_average_time = inference_time / real_batch ;
  // MBLOG_INFO << "inference time: " << inference_time << " average time: " << inference_average_time; 
  return Build_Batch_Outputs(data_ctx);
  // return Build_Outputs(data_ctx);
}

modelbox::Status modelbox::RKNPU2Inference::Deinit() {
  std::lock_guard<std::mutex> lk(rknpu2_infer_mtx_);
  if (ctx_ != 0) {
    // 发现，ctrlc退出的时候 rknn_destroy之前需要等一下，
    // 有可能3568比较慢，不然会导致下一次推理异常
    usleep(1000);
    rknn_destroy(ctx_);
    ctx_ = 0;
  }

  return modelbox::STATUS_SUCCESS;
}
