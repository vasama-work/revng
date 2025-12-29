#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/DoxygenEmitter.h"

namespace ptml {

using CDoxygenCommentEmitter = //
  DoxygenCommentEmitter<CTokenEmitter::CommentEmitter>;

[[nodiscard]] inline CDoxygenCommentEmitter
emitDoxygenLineComment(CTokenEmitter &CE) {
  return CDoxygenCommentEmitter({ .LinePrefix = "/ " },
                                CE,
                                CTokenEmitter::CommentKind::Line);
}

[[nodiscard]] inline CDoxygenCommentEmitter
emitDoxygenBlockComment(CTokenEmitter &CE) {
  DoxygenCommentConfiguration Configuration = {
    .CommentHeader = "*",
    .CommentFooter = " ",
    .LinePrefix = " * ",
  };

  return CDoxygenCommentEmitter(Configuration,
                                CE,
                                CTokenEmitter::CommentKind::Block);
}

} // namespace ptml
