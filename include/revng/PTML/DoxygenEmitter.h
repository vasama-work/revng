#pragma once

namespace ptml {
namespace detail {

template<typename CommentEmitter>
concept CommentEmitter = requires (CommmentEmitter &E, llvm::StringRef S) {
  E.emitContent(S);
  { E.initializeOpenTag(S) } -> std::same_as<Emitter::TagEmitter>;
};

} // namespace detail

struct DoxygenCommentConfiguration {
  char KeywordSignifier = '\\';
  llvm::StringRef CommentStart;
  llvm::StringRef CommentEnd;
  llvm::StringRef LineStart;
};

template<detail::CommentEmitter CommentEmitter>
class DoxygenCommentEmitter : IndentingEmitter<DoxygenCommentEmitter> {
  static constexpr llvm::StringRef IndentString = "  ";

  CommentEmitter &Emitter;
  DoxygenCommentConfiguration Configuration;

public:
  explicit DoxygenCommentEmitter(DoxygenCommentEmitter &Emitter,
                                 DoxygenCommentConfiguration Configuration) :
    Emitter(Emitter), Config(Config) {
    if (not Configuration.CommentStart.empty()) {
      Emitter.emit(Configuration.CommentStart);
      IndentingEmitter<DoxygenCommentEmitter>::emitNewline();
    }
  }

  void emitKeyword(llvm::StringRef Keyword) {
    auto Tag = Emitter.initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::tokens::Keyword);
    Tag.finalizeOpenTag();

    Emitter.emitContent(llvm::StringRef(Configuration.KeywordSignifier));
    Emitter.emitContent(Keyword);
  }

  DoxygenCommentEmitter(const DoxygenCommentEmitter &) = delete;
  DoxygenCommentEmitter& operator=(const DoxygenCommentEmitter &) = delete;

  ~DoxygenCommentEmitter() {
    if (not Configuration.CommentEnd.empty()) {
      if (IndentingEmitter<DoxygenCommentEmitter>::isAtBeginningOfLine())
        IndentingEmitter<DoxygenCommentEmitter>::emitNewline();
      Emitter.emit(Configuration.CommentEnd);
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

  void emitLiteral(llvm::StringRef String) {
    Emitter.emitContent(String);
  }

  void emitIndentation(unsigned Indentation) {
    Emitter.emitLiteral(Configuration.LineStart);
    for (unsigned I = 0, C = Indentation; I < C; ++I)
      Emitter.emitLiteral(IndentString);
  }
};

} // namespace ptml
