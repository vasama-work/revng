#pragma once

#include "llvm/ADT/StringRef.h"

namespace mlir::clift {

// WIP: This should probably moved somewhere else:
inline std::string sanitizeIdentifier(llvm::StringRef Identifier) {
  static constexpr auto IsIdentifierInitialChar = [](char X) {
    if (X == '_')
      return true;
    if ('a' <= X and X <= 'z')
      return true;
    if ('A' <= X and X <= 'Z')
      return true;
    return false;
  };

  static constexpr auto IsIdentifierChar = [](char X) {
    if (IsIdentifierInitialChar(X))
      return true;
    if ('0' <= X and X <= '9')
      return true;
    return false;
  };

  std::string SanitaryIdentifier = Identifier.str();

  for (char &X : SanitaryIdentifier) {
    if (not IsIdentifierChar(X))
      X = '_';
  }

  if (not SanitaryIdentifier.empty()
      and not IsIdentifierInitialChar(SanitaryIdentifier.front()))
    SanitaryIdentifier.insert(SanitaryIdentifier.begin(), '_');

  return SanitaryIdentifier;
}

} // namespace mlir::clift
