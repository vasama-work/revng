#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Support/Assert.h"

namespace ptml {

template<typename EmitterT, typename TraitsT>
class IndentingEmitter : protected EmitterT {
  unsigned Indentation = 0;
  bool IsAtBeginningOfLine = true;

public:
  template<typename... ArgsT>
    requires std::constructible_from<EmitterT, ArgsT...>
  explicit IndentingEmitter(ArgsT &&...Args) :
    EmitterT(std::forward<ArgsT>(Args)...) {}

  void indent(int Offset) {
    revng_assert(Offset >= 0 or static_cast<unsigned>(-Offset) <= Indentation,
                 "Offset would result in negative indentation.");

    Indentation += static_cast<unsigned>(Offset);
  }

  [[nodiscard]] unsigned indentation() const { return Indentation; }

  [[nodiscard]] bool isAtBeginningOfLine() const { return IsAtBeginningOfLine; }

  void emit(llvm::StringRef String) {
    if (not String.empty()) {
      for (auto [I, R] : llvm::enumerate(std::views::split(String, '\n'))) {
        llvm::StringRef Line = std::string_view(R.begin(), R.end());

        if (I != 0)
          emitNewline();

        if (not Line.empty()) {
          emitIndentationIfNeeded();
          EmitterT::emit(Line);
        }
      }

      IsAtBeginningOfLine = String.back() == '\n';
    }
  }

  void emitNewline() {
    EmitterT::emit(llvm::StringRef("\n"));
    IsAtBeginningOfLine = true;
  }

protected:
  void emitIndentationIfNeeded() {
    if (IsAtBeginningOfLine) {
      IsAtBeginningOfLine = false;
      TraitsT::emitIndentation(static_cast<EmitterT &>(*this), Indentation);
    }
  }
};

} // namespace ptml
