//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Constants.h"
#include "revng/PTML/Emitter.h"

using namespace ptml;
using ptml::detail::EmitterBase;
using ptml::detail::BasicTagEmitter;

static constexpr llvm::StringRef IndentString = "  ";

// PTML requires escaping some characters. Currently we escape angle brackets
// and ampersands unconditionally. Quotes are escaped only within attribute
// values, which are themselves delimited by quotes. Attribute values delimited
// by apostrophes are not emitted, so there is no need to ever escape them.
//
// In some situations escaping angle brackets could be avoided, but these
// situations are either not encountered in practice or introduce asymmetries.
// For this reason they are escaped unconditionally.

template<bool EscapeQuotes = false>
static bool requiresEscaping(char Character) {
  switch (Character) {
  case '<':
  case '>':
  case '&':
    return true;
  case '\"':
    return EscapeQuotes;
  default:
    return false;
  }
}

static llvm::StringRef getEscape(char Character) {
  switch (Character) {
  case '\"':
    return "&quot;";
  case '<':
    return "&lt;";
  case '>':
    return "&gt;";
  case '&':
    return "&amp;";
  default:
    revng_abort("The specified character does not require escaping.");
  }
}

template<bool EscapeQuotes, typename EmitT>
static void emitEscapedImpl(llvm::StringRef String, EmitT Emit) {
  auto Begin = String.data();
  auto End = Begin + String.size();

  while (Begin != End) {
    auto Pos = std::find_if(Begin, End, [](char Character) {
      return requiresEscaping<EscapeQuotes>(Character);
    });

    Emit(llvm::StringRef(std::string_view(Begin, Pos)));

    if (Pos != End)
      Emit(getEscape(*Pos++));

    Begin = Pos;
  }
}

//===------------------------- BasicPTMLTagEmitter ------------------------===//

template<typename E>
void BasicPTMLTagEmitter<E>::initializeOpenTagImpl(E &ParentEmitter,
                                                   llvm::StringRef Tag) {
  ParentEmitter.enterOpenTag(*this);

  this->ParentEmitter = &ParentEmitter;
  this->Tag = Tag;
  this->IsOpenTagFinalized = false;

  if (ParentEmitter.EmitTags)
    ParentEmitter.OS << '<' << Tag;
}

template<typename E>
void TagEmitter<E>::emitAttributeImpl(llvm::StringRef Name,
                                             llvm::StringRef Value) {
  revng_assert(ParentEmitter->CurrentOpenTagEmitter == this);

  if (ParentEmitter->EmitTags) {
    ParentEmitter->OS << ' ' << Name << '=' << '"';
    emitAttributeValue(Value);
    ParentEmitter->OS << '"';
  }
}

template<typename E>
void TagEmitter<E>::emitListAttributeImpl(llvm::StringRef Name,
                                          llvm::ArrayRef<llvm::StringRef> Values) {
  revng_assert(ParentEmitter->CurrentOpenTagEmitter == this);

  if (ParentEmitter->EmitTags) {
    ParentEmitter->OS << ' ' << Name << '=' << '"';

    bool InsertComma = false;
    for (auto [I, Value] : llvm::enumerate(Values)) {
      revng_assert(not Value.contains(','),
                   "List attribute values shall not contain commas.");

      if (I == 0)
        ParentEmitter->OS << ',';

      emitAttributeValue(Value);
    }

    ParentEmitter->OS << '"';
  }
}

template<typename E>
void TagEmitter<E>::finalizeOpenTagImpl() {
  if (not IsOpenTagFinalized) {
    if (ParentEmitter->EmitTags)
      ParentEmitter->OS << '>';

    IsOpenTagFinalized = true;
    ParentEmitter.leaveOpenTag(*this);
  }
}

template<typename E>
void TagEmitter<E>::closeImpl() {
  if (ParentEmitter != nullptr) {
    finalizeOpenTagImpl();

    if (ParentEmitter->EmitTags)
      ParentEmitter->OS << '<' << '/' << Tag << '>';
  }
  ParentEmitter = nullptr;
}

//===-------------------------- SimplePTMLEmitter -------------------------===//

void SimplePTMLEmitter::emitContent(llvm::StringRef Content) {
  revng_assert(CurrentOpenTagEmitter == nullptr,
               "Cannot emit content while an unfinalized TagEmitter is "
               "associated with this emitter.");

  emitEscaped(Content);
}

template<bool EscapeQuotes>
void SimplePTMLEmitter::emitEscaped(llvm::StringRef String) {
  emitEscapedImpl<EscapeQuotes>(String, [this](llvm::StringRef S) {
    OS << S;
  });
}

template class BasicPTMLTagEmitter<SimplePTMLEmitter>;

//===------------------------ IndentingPTMLEmitter ------------------------===//

void detail::PTMLIndentationTraits::emitIndentation(SimplePTMLEmitter &Emitter,
                                                    unsigned Indentation) {
  if (Indentation != 0) {
    SimplePTMLEmitter::TagEmitter Tag;

    if (EmitTags) {
      Tag.initializeOpenTag(*this, ptml::tags::Span);
      Tag.emitAttribute(ptml::attributes::Token, ptml::tokens::Indentation);
      Tag.finalizeOpenTag();
    }

    for (unsigned I = 0, C = Indentation; I < C; ++I)
      OS << IndentString;
  }
}

void IndentingPTMLEmitter::emitContent(llvm::StringRef String) {
  revng_assert(CurrentOpenTagEmitter == nullptr,
               "Cannot emit content while an unfinalized TagEmitter is "
               "associated with this emitter.");

  emitEscaped(String);
}

template<bool EscapeQuotes>
void Emitter::emitEscaped(llvm::StringRef String) {
  emitEscapedImpl<EscapeQuotes>(String, [this](llvm::StringRef Part) {
    IndentingEmitter::emit(Part);
  });
}

void IndentingPTMLEmitter::enterOpenTag(const PTMLTagEmitterBase &TagEmitter) {
  if (EmitTags)
    emitIndentationIfNeeded();

  SimplePTMLEmitter::enterOpenTag(TagEmitter);
}

template class BasicPTMLTagEmitter<IndentingPTMLEmitter>;
