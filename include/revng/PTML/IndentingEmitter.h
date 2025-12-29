#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Support/Assert.h"

namespace ptml {

template<typename Derived>
class IndentingEmitter {
  unsigned Indentation = 0;
  bool IsAtBeginningOfLine = true;

public:
  void indent(int Offset) {
    if (Offset < 0)
      revng_assert(Indentation >= static_cast<unsigned>(-Offset));

    Indentation += static_cast<unsigned>(Offset);
  }

  [[nodiscard]] unsigned indentation() const {
    return Indentation;
  }

  [[nodiscard]] bool isAtBeginningOfLine() const {
    return IsAtBeginningOfLine;
  }

  void emit(llvm::StringRef String) {
    for (auto [I, R] : llvm::enumerate(std::views::split(String, '\n'))) {
      if (I != 0)
        emitNewline();

      emitLiteralImpl(std::string_view(R.begin(), R.end()));
    }

    IsAtBeginningOfLine = String.back() == '\n';
  }

  void emitLiteral(llvm::StringRef String) {
    if (not String.empty()) {
      revng_assert(not String.contains('\n'));
      emitLiteralImpl(String);
    }
  }

  void emitNewline() {
    static_cast<Derived *>(this)->emitLiteral("\n");
    IsAtBeginningOfLine = true;
  }

private:
  void emitLiteralImpl(llvm::StringRef String) {
    revng_assert(not String.empty());
    if (IsAtBeginningOfLine) {
      IsAtBeginningOfLine = false;
      static_cast<Derived *>(this)->emitIndentation(Indentation);
    }
    static_cast<Derived *>(this)->emitLiteral(String);
  }
};

} // namespace ptml
