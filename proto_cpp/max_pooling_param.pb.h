// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: max_pooling_param.proto

#ifndef PROTOBUF_max_5fpooling_5fparam_2eproto__INCLUDED
#define PROTOBUF_max_5fpooling_5fparam_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3004000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3004000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "size.pb.h"
// @@protoc_insertion_point(includes)
class XZY_MaxPoolingParam;
class XZY_MaxPoolingParamDefaultTypeInternal;
extern XZY_MaxPoolingParamDefaultTypeInternal _XZY_MaxPoolingParam_default_instance_;

namespace protobuf_max_5fpooling_5fparam_2eproto {
// Internal implementation detail -- do not call these.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[];
  static const ::google::protobuf::uint32 offsets[];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static void InitDefaultsImpl();
};
void AddDescriptors();
void InitDefaults();
}  // namespace protobuf_max_5fpooling_5fparam_2eproto

// ===================================================================

class XZY_MaxPoolingParam : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:XZY_MaxPoolingParam) */ {
 public:
  XZY_MaxPoolingParam();
  virtual ~XZY_MaxPoolingParam();

  XZY_MaxPoolingParam(const XZY_MaxPoolingParam& from);

  inline XZY_MaxPoolingParam& operator=(const XZY_MaxPoolingParam& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  XZY_MaxPoolingParam(XZY_MaxPoolingParam&& from) noexcept
    : XZY_MaxPoolingParam() {
    *this = ::std::move(from);
  }

  inline XZY_MaxPoolingParam& operator=(XZY_MaxPoolingParam&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const XZY_MaxPoolingParam& default_instance();

  static inline const XZY_MaxPoolingParam* internal_default_instance() {
    return reinterpret_cast<const XZY_MaxPoolingParam*>(
               &_XZY_MaxPoolingParam_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(XZY_MaxPoolingParam* other);
  friend void swap(XZY_MaxPoolingParam& a, XZY_MaxPoolingParam& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline XZY_MaxPoolingParam* New() const PROTOBUF_FINAL { return New(NULL); }

  XZY_MaxPoolingParam* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const XZY_MaxPoolingParam& from);
  void MergeFrom(const XZY_MaxPoolingParam& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(XZY_MaxPoolingParam* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated int32 col_table = 3;
  int col_table_size() const;
  void clear_col_table();
  static const int kColTableFieldNumber = 3;
  ::google::protobuf::int32 col_table(int index) const;
  void set_col_table(int index, ::google::protobuf::int32 value);
  void add_col_table(::google::protobuf::int32 value);
  const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
      col_table() const;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
      mutable_col_table();

  // .XZY_Size col_matrix_size = 1;
  bool has_col_matrix_size() const;
  void clear_col_matrix_size();
  static const int kColMatrixSizeFieldNumber = 1;
  const ::XZY_Size& col_matrix_size() const;
  ::XZY_Size* mutable_col_matrix_size();
  ::XZY_Size* release_col_matrix_size();
  void set_allocated_col_matrix_size(::XZY_Size* col_matrix_size);

  // .XZY_Size kernel_size = 2;
  bool has_kernel_size() const;
  void clear_kernel_size();
  static const int kKernelSizeFieldNumber = 2;
  const ::XZY_Size& kernel_size() const;
  ::XZY_Size* mutable_kernel_size();
  ::XZY_Size* release_kernel_size();
  void set_allocated_kernel_size(::XZY_Size* kernel_size);

  // @@protoc_insertion_point(class_scope:XZY_MaxPoolingParam)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 > col_table_;
  mutable int _col_table_cached_byte_size_;
  ::XZY_Size* col_matrix_size_;
  ::XZY_Size* kernel_size_;
  mutable int _cached_size_;
  friend struct protobuf_max_5fpooling_5fparam_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// XZY_MaxPoolingParam

// .XZY_Size col_matrix_size = 1;
inline bool XZY_MaxPoolingParam::has_col_matrix_size() const {
  return this != internal_default_instance() && col_matrix_size_ != NULL;
}
inline void XZY_MaxPoolingParam::clear_col_matrix_size() {
  if (GetArenaNoVirtual() == NULL && col_matrix_size_ != NULL) delete col_matrix_size_;
  col_matrix_size_ = NULL;
}
inline const ::XZY_Size& XZY_MaxPoolingParam::col_matrix_size() const {
  const ::XZY_Size* p = col_matrix_size_;
  // @@protoc_insertion_point(field_get:XZY_MaxPoolingParam.col_matrix_size)
  return p != NULL ? *p : *reinterpret_cast<const ::XZY_Size*>(
      &::_XZY_Size_default_instance_);
}
inline ::XZY_Size* XZY_MaxPoolingParam::mutable_col_matrix_size() {
  
  if (col_matrix_size_ == NULL) {
    col_matrix_size_ = new ::XZY_Size;
  }
  // @@protoc_insertion_point(field_mutable:XZY_MaxPoolingParam.col_matrix_size)
  return col_matrix_size_;
}
inline ::XZY_Size* XZY_MaxPoolingParam::release_col_matrix_size() {
  // @@protoc_insertion_point(field_release:XZY_MaxPoolingParam.col_matrix_size)
  
  ::XZY_Size* temp = col_matrix_size_;
  col_matrix_size_ = NULL;
  return temp;
}
inline void XZY_MaxPoolingParam::set_allocated_col_matrix_size(::XZY_Size* col_matrix_size) {
  delete col_matrix_size_;
  col_matrix_size_ = col_matrix_size;
  if (col_matrix_size) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:XZY_MaxPoolingParam.col_matrix_size)
}

// .XZY_Size kernel_size = 2;
inline bool XZY_MaxPoolingParam::has_kernel_size() const {
  return this != internal_default_instance() && kernel_size_ != NULL;
}
inline void XZY_MaxPoolingParam::clear_kernel_size() {
  if (GetArenaNoVirtual() == NULL && kernel_size_ != NULL) delete kernel_size_;
  kernel_size_ = NULL;
}
inline const ::XZY_Size& XZY_MaxPoolingParam::kernel_size() const {
  const ::XZY_Size* p = kernel_size_;
  // @@protoc_insertion_point(field_get:XZY_MaxPoolingParam.kernel_size)
  return p != NULL ? *p : *reinterpret_cast<const ::XZY_Size*>(
      &::_XZY_Size_default_instance_);
}
inline ::XZY_Size* XZY_MaxPoolingParam::mutable_kernel_size() {
  
  if (kernel_size_ == NULL) {
    kernel_size_ = new ::XZY_Size;
  }
  // @@protoc_insertion_point(field_mutable:XZY_MaxPoolingParam.kernel_size)
  return kernel_size_;
}
inline ::XZY_Size* XZY_MaxPoolingParam::release_kernel_size() {
  // @@protoc_insertion_point(field_release:XZY_MaxPoolingParam.kernel_size)
  
  ::XZY_Size* temp = kernel_size_;
  kernel_size_ = NULL;
  return temp;
}
inline void XZY_MaxPoolingParam::set_allocated_kernel_size(::XZY_Size* kernel_size) {
  delete kernel_size_;
  kernel_size_ = kernel_size;
  if (kernel_size) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:XZY_MaxPoolingParam.kernel_size)
}

// repeated int32 col_table = 3;
inline int XZY_MaxPoolingParam::col_table_size() const {
  return col_table_.size();
}
inline void XZY_MaxPoolingParam::clear_col_table() {
  col_table_.Clear();
}
inline ::google::protobuf::int32 XZY_MaxPoolingParam::col_table(int index) const {
  // @@protoc_insertion_point(field_get:XZY_MaxPoolingParam.col_table)
  return col_table_.Get(index);
}
inline void XZY_MaxPoolingParam::set_col_table(int index, ::google::protobuf::int32 value) {
  col_table_.Set(index, value);
  // @@protoc_insertion_point(field_set:XZY_MaxPoolingParam.col_table)
}
inline void XZY_MaxPoolingParam::add_col_table(::google::protobuf::int32 value) {
  col_table_.Add(value);
  // @@protoc_insertion_point(field_add:XZY_MaxPoolingParam.col_table)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
XZY_MaxPoolingParam::col_table() const {
  // @@protoc_insertion_point(field_list:XZY_MaxPoolingParam.col_table)
  return col_table_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
XZY_MaxPoolingParam::mutable_col_table() {
  // @@protoc_insertion_point(field_mutable_list:XZY_MaxPoolingParam.col_table)
  return &col_table_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)


// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_max_5fpooling_5fparam_2eproto__INCLUDED
