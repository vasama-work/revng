#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/PTML/IndentingEmitter.h"

namespace ptml {

template<typename EmitterT>
concept Emitter = requires (EmitterT &Emitter, llvm::StringRef String) {
  // void emit(llvm::StringRef String);
  Emitter.emit(String);
};

enum class Tagging : bool {
  Disabled,
  Enabled,
};

class IndentingPTMLEmitter;

namespace detail {

class PTMLTagEmitterBase {};

} // namespace detail

/// \brief RAII object used for emitting PTML tags.
///
/// BasicPTMLTagEmitter has three states:
/// 1. Uninitialized:
///    The BasicPTMLTagEmitter has been default-constructed or initialised and
///    subsequently closed.
///
/// 2. Initialized:
///    Emission of the opening tag has been started, but not yet completed. It
///    is only in this state that emission of attributes is possible, while at
///    the same time emission of content via the associated ptml::Emitter is
///    disallowed.
///
///    This state can be entered using the initializeOpenTag member functions
///    of either Emitter or BasicPTMLTagEmitter, or by constructing a BasicPTMLTagEmitter using
///    its non-default constructor.
///
///    Emitter::initializeOpenTag is a convenience function which returns a
///    BasicPTMLTagEmitter in the initialized state.
///
/// 3. Finalized:
///    Emission of the opening tag has been completed, but the closing tag has
///    not yet been emitted. In this state tag content can be emitted via the
///    associated ptml::Emitter.
///
///    This state is entered using the finalizeOpenTag member function.
///
/// The closing tag is emitted by explicitly calling the close member function,
/// or implicitly by the destructor, which also takes care of finalizing the
/// open tag if necessary.
///
/// At any given time, the emitter may be associated with multiple TagEmitters
/// but only the innermost can have an unfinalized open tag.
template<typename EmitterT>
class BasicPTMLTagEmitter : public detail::PTMLTagEmitterBase {
  EmitterT *ParentEmitter;
  llvm::StringRef Tag;
  bool IsOpenTagFinalized = false;

public:
  using EmitterType = EmitterT;

  BasicPTMLTagEmitter() : ParentEmitter(nullptr) {}

  explicit BasicPTMLTagEmitter(EmitterT &ParentEmitter, llvm::StringRef Tag) {
    revng_assert(ParentEmitter.CurrentOpenTagEmitter == nullptr,
                 "The parent Emitter is already associated with an "
                 "unfinalized BasicPTMLTagEmitter.");

    initializeOpenTagImpl(ParentEmitter, Tag);
  }

  BasicPTMLTagEmitter(const BasicPTMLTagEmitter &) = delete;
  BasicPTMLTagEmitter &operator=(const BasicPTMLTagEmitter &) = delete;

  ~BasicPTMLTagEmitter() { closeImpl(); }

  BasicPTMLTagEmitter &initializeOpenTag(EmitterT &ParentEmitter,
                                         llvm::StringRef Tag) & {
    revng_assert(this->ParentEmitter == nullptr,
                 "This BasicPTMLTagEmitter was already initialized.");
    revng_assert(ParentEmitter.CurrentOpenTagEmitter == nullptr,
                 "The parent Emitter is already associated with an "
                 "unfinalized BasicPTMLTagEmitter.");

    initializeOpenTagImpl(ParentEmitter, Tag);
    return *this;
  }

  BasicPTMLTagEmitter &emitAttribute(llvm::StringRef Name,
                                     llvm::StringRef Value) & {
    revng_assert(ParentEmitter != nullptr,
                 "This BasicPTMLTagEmitter has not been initialized.");
    revng_assert(not IsOpenTagFinalized,
                 "This BasicPTMLTagEmitter has already been finalized.");

    emitAttributeImpl(Name, Value);
    return *this;
  }

  BasicPTMLTagEmitter &
  emitListAttribute(llvm::StringRef Name,
                    llvm::ArrayRef<llvm::StringRef> Values) & {
    revng_assert(ParentEmitter != nullptr,
                 "This BasicPTMLTagEmitter has not been initialized.");
    revng_assert(not IsOpenTagFinalized,
                 "This BasicPTMLTagEmitter has already been finalized.");

    emitListAttributeImpl(Name, Values);
    return *this;
  }

  [[nodiscard]] bool isOpenTagFinalized() const { return IsOpenTagFinalized; }

  void finalizeOpenTag() {
    revng_assert(ParentEmitter != nullptr,
                 "This BasicPTMLTagEmitter has not been initialized.");
    revng_assert(not IsOpenTagFinalized,
                 "This BasicPTMLTagEmitter has already been finalized.");
    finalizeOpenTagImpl();
  }

  [[nodiscard]] bool isOpen() const { return ParentEmitter != nullptr; }

  void close() {
    revng_assert(ParentEmitter != nullptr,
                 "This BasicPTMLTagEmitter has not been initialized.");
    closeImpl();
  }

private:
  void initializeOpenTagImpl(EmitterT &ParentEmitter, llvm::StringRef Tag);

  void emitAttributeImpl(llvm::StringRef Name, llvm::StringRef Value);
  void emitListAttributeImpl(llvm::StringRef Name,
                             llvm::ArrayRef<llvm::StringRef> Values);

  void finalizeOpenTagImpl();
  void closeImpl();
};

namespace detail {

class PTMLEmitterBase {
public: // WIP
  const PTMLTagEmitterBase *CurrentOpenTagEmitter = nullptr;

protected:
  bool isEmittingOpenTag() const {
    return CurrentOpenTagEmitter != nullptr;
  }

  void enterOpenTag(const PTMLTagEmitterBase &TagEmitter) {
    revng_assert(CurrentOpenTagEmitter == nullptr);
    CurrentOpenTagEmitter = &TagEmitter;
  }

  void leaveOpenTag(const PTMLTagEmitterBase &TagEmitter) {
    revng_assert(CurrentOpenTagEmitter == &TagEmitter);
    CurrentOpenTagEmitter = nullptr;
  }

  template<typename EmitterT>
  friend class BasicPTMLTagEmitter;
};

} // namespace detail

template<typename TagEmitterT>
concept PTMLTagEmitter = requires (TagEmitterT &TagEmitter, llvm::StringRef String, llvm::ArrayRef<llvm::StringRef> Array) {
  typename TagEmitterT::EmitterType;

  // TagEmitter &initializeOpenTag(llvm::StringRef Tag);
  { TagEmitter.initializeOpenTag(String) } -> std::same_as<TagEmitterT &>;

  // TagEmitter &emitAttribute(llvm::StringRef Tag, llvm::StringRef Value);
  { TagEmitter.emitAttribute(String, String) } -> std::same_as<TagEmitterT &>;

  // TagEmitter &emitListAttribute(llvm::StringRef Tag,
  //                               llvm::ArrayRef<llvm::StringRef> Values);
  { TagEmitter.emitListAttribute(String, Array) } -> std::same_as<TagEmitterT &>;

  // bool isOpenTagFinalized() const;
  { std::as_const(TagEmitter).isOpenTagFinalized() } -> std::same_as<bool>;

  // void finalizeOpenTag();
  TagEmitter.finalizeOpenTag();

  // bool isOpen() const;
  { std::as_const(TagEmitter).isOpen() } -> std::same_as<bool>;

  // void close();
  TagEmitter.close();
};

template<typename EmitterT>
concept PTMLEmitter = Emitter<EmitterT> and requires (EmitterT &Emitter, llvm::StringRef String) {
  // using TagEmitter = ...;
  requires PTMLTagEmitter<typename EmitterT::TagEmitter>;

  typename EmitterT::TagEmitter(Emitter, String);

  // TagEmitter initializeOpenTag(llvm::StringRef String);
  {
    Emitter.initializeOpenTag(String)
  } -> std::same_as<typename EmitterT::TagEmitter>;
};

/// \brief Provides a streaming interface for emitting PTML tags and content.
///
/// Underlying byte-IO is done via the provided llvm::raw_ostream reference.
///
/// PTML tag emission is performed using Emitter::BasicPTMLTagEmitter, which is an RAII
/// object guaranteeing emission of well-formed PTML. Tag content is emitted
/// using the Emitter interface. See the documentation of Emitter::BasicPTMLTagEmitter
/// for more information.
///
/// PTML tag emission can be toggled using the ptml::Tagging parameter. Note
/// that valid usage of the PTML tag emission interface is checked regardless
/// of whether PTML tag emission is enabled.
///
/// As of the introduction of this class, there are no known use cases for it.
/// It was only introduced to avoid reentrancy in IndentingPTMLEmitter.
class SimplePTMLEmitter : detail::PTMLEmitterBase {
  llvm::raw_ostream &OS;
  bool EmitTags = false;

public:
  using TagEmitter = BasicPTMLTagEmitter<SimplePTMLEmitter>;

  explicit SimplePTMLEmitter(llvm::raw_ostream &OS, Tagging Tags) :
    OS(OS), EmitTags(Tags == Tagging::Enabled) {}

  void emit(llvm::StringRef Content);

  [[nodiscard]] TagEmitter initializeOpenTag(llvm::StringRef Tag) {
    return TagEmitter(*this, Tag);
  }

protected:
  template<bool EscapeQuotes = false>
  void emitEscaped(llvm::StringRef String);

  friend IndentingPTMLEmitter;

  template<typename EmitterT>
  friend class BasicPTMLTagEmitter;
};

extern template class BasicPTMLTagEmitter<SimplePTMLEmitter>;

namespace detail {

struct PTMLIndentationTraits {
  void emitIndentation(SimplePTMLEmitter &Emitter, unsigned Indentation);
};

} // namespace detail

/// \brief Provides a streaming interface for emitting PTML tags and content
///        with automatic indentation.
class IndentingPTMLEmitter : IndentingEmitter<SimplePTMLEmitter,
                                              detail::PTMLIndentationTraits> {
public:
  using TagEmitter = BasicPTMLTagEmitter<IndentingPTMLEmitter>;

  explicit IndentingPTMLEmitter(llvm::raw_ostream &OS, Tagging Tags) :
    IndentingEmitter(OS, Tags) {}

  using IndentingEmitter::indent;
  using IndentingEmitter::indentation;
  using IndentingEmitter::isAtBeginningOfLine;

  void emit(llvm::StringRef String);

  [[nodiscard]] TagEmitter initializeOpenTag(llvm::StringRef Tag) {
    return TagEmitter(*this, Tag);
  }

private:
  template<bool EscapeQuotes = false>
  void emitEscaped(llvm::StringRef String);

  void enterOpenTag(const detail::PTMLTagEmitterBase &TagEmitter);

  template<typename EmitterT>
  friend class BasicPTMLTagEmitter;
};

extern template class BasicPTMLTagEmitter<IndentingPTMLEmitter>;

} // namespace ptml
