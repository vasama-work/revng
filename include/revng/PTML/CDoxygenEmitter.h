#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/DoxygenEmitter.h"

namespace ptml {
namespace detail {

// Short enough to fit within the column limit...
using CDoxComEmitter = DoxygenCommentEmitter<CTokenEmitter::CommentEmitter>;

} // namespace detail

using CDoxygenCommentEmitter = detail::CDoxComEmitter;

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
