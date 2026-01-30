#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Constants.h"

namespace ptml {

struct DoxygenCommentConfiguration {
  char KeywordSignifier = '\\';
  std::optional<llvm::StringRef> CommentHeader;
  std::optional<llvm::StringRef> CommentFooter;
  llvm::StringRef LinePrefix;
};

namespace detail {

struct DoxygenIndentationTraits;

template<typename EmitterT>
class DoxygenEmitterBase : protected EmitterT {
protected:
  DoxygenCommentConfiguration Configuration;

  template<typename... ArgsT>
    requires std::constructible_from<EmitterT, ArgsT...>
  explicit DoxygenEmitterBase(const DoxygenCommentConfiguration &Configuration,
                              ArgsT &&...Args) :
    EmitterT(std::forward<ArgsT>(Args)...), Configuration(Configuration) {}

  friend DoxygenIndentationTraits;
};

struct DoxygenIndentationTraits {
  template<Emitter EmitterT>
  void emitIndentation(DoxygenEmitterBase<EmitterT> &Emitter,
                       unsigned Indentation) {
    Emitter.emit(Emitter.Configuration.LinePrefix);
    for (unsigned I = 0, C = Indentation; I < C; ++I)
      Emitter.emit("  ");
  }
};

} // namespace detail

template<PTMLEmitter EmitterT>
class DoxygenEmitter
  : IndentingEmitter<EmitterT, detail::DoxygenIndentationTraits> {

  using BaseType = IndentingEmitter<EmitterT, detail::DoxygenIndentationTraits>;

  DoxygenCommentConfiguration Configuration;

public:
  template<typename... ArgsT>
    requires std::constructible_from<EmitterT, ArgsT...>
  explicit DoxygenEmitter(const DoxygenCommentConfiguration &Configuration,
                          ArgsT &&...Args) :
    BaseType(Configuration, std::forward<ArgsT>(Args)...) {
    if (Configuration.CommentHeader) {
      EmitterT::emit(*Configuration.CommentHeader);
      BaseType::emit("\n");
    }
  }

  void emitKeyword(llvm::StringRef Keyword) {
    auto Tag = EmitterT::initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::doxygen::tokens::Keyword);
    Tag.finalizeOpenTag();

    BaseType::emit(llvm::StringRef(&Configuration.KeywordSignifier, 1));
    BaseType::emit(Keyword);
  }

  DoxygenEmitter(const DoxygenEmitter &) = delete;
  DoxygenEmitter &operator=(const DoxygenEmitter &) = delete;

  ~DoxygenEmitter() {
    if (Configuration.CommentFooter) {
      if (not BaseType::isAtBeginningOfLine())
        BaseType::emit("\n");
      EmitterT::emit(*Configuration.CommentFooter);
    }
  }

  using BaseType::emit;
};

} // namespace ptml
