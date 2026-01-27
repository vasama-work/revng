#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/CommentEmitter.h"
#include "revng/PTML/Constants.h"

namespace ptml {

struct DoxygenCommentConfiguration {
  char KeywordSignifier = '\\';
  std::optional<llvm::StringRef> CommentHeader;
  std::optional<llvm::StringRef> CommentFooter;
  llvm::StringRef LinePrefix;
};

template<CommentEmitter CommentEmitterT>
class DoxygenCommentEmitter
  : IndentingEmitter<DoxygenCommentEmitter<CommentEmitterT>> {

  friend IndentingEmitter<DoxygenCommentEmitter<CommentEmitterT>>;

  static constexpr llvm::StringRef IndentString = "  ";

  CommentEmitterT Emitter;
  DoxygenCommentConfiguration Configuration;

public:
  template<typename... ArgsT>
    requires std::constructible_from<CommentEmitterT, ArgsT...>
  explicit DoxygenCommentEmitter(DoxygenCommentConfiguration Configuration,
                                 ArgsT &&...Args) :
    Emitter(std::forward<ArgsT>(Args)...), Configuration(Configuration) {
    if (Configuration.CommentHeader) {
      Emitter.emitContent(*Configuration.CommentHeader);
      IndentingEmitter<DoxygenCommentEmitter>::emitNewline();
    }
  }

  void emitKeyword(llvm::StringRef Keyword) {
    auto Tag = Emitter.initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::doxygen::tokens::Keyword);
    Tag.finalizeOpenTag();

    llvm::StringRef Signifier(&Configuration.KeywordSignifier, 1);
    IndentingEmitter<DoxygenCommentEmitter>::emit(Signifier);
    IndentingEmitter<DoxygenCommentEmitter>::emit(Keyword);
  }

  DoxygenCommentEmitter(const DoxygenCommentEmitter &) = delete;
  DoxygenCommentEmitter &operator=(const DoxygenCommentEmitter &) = delete;

  ~DoxygenCommentEmitter() {
    if (Configuration.CommentFooter) {
      if (not IndentingEmitter<DoxygenCommentEmitter>::isAtBeginningOfLine())
        IndentingEmitter<DoxygenCommentEmitter>::emitNewline();
      Emitter.emitContent(*Configuration.CommentFooter);
    }
  }

  void emitContent(llvm::StringRef Content) {
    IndentingEmitter<DoxygenCommentEmitter>::emit(Content);
  }

  void emitContentNewline() {
    IndentingEmitter<DoxygenCommentEmitter>::emitNewline();
  }

private:
  //===-------------------- IndentingEmitter interface --------------------===//

  void emitLiteral(llvm::StringRef String) { Emitter.emitContent(String); }

  void emitIndentation(unsigned Indentation) {
    Emitter.emitContent(Configuration.LinePrefix);
    for (unsigned I = 0, C = Indentation; I < C; ++I)
      Emitter.emitContent(IndentString);
  }
};

} // namespace ptml
