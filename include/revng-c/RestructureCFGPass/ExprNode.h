#ifndef REVNGC_RESTRUCTURE_CFG_EXPRNODE_H
#define REVNGC_RESTRUCTURE_CFG_EXPRNODE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>

// LLVM includes
#include <llvm/Support/Casting.h>

class ExprNode {
public:
  enum NodeKind { NK_Atomic,
                  NK_Not,
                  NK_And,
                  NK_Or };

private:
  const NodeKind Kind;

public:
  NodeKind getKind() const { return Kind; }

  ExprNode(NodeKind K) : Kind(K) {}

};

class AtomicNode : public ExprNode {
private:
  llvm::BasicBlock *ConditionBB;

public:
  AtomicNode(llvm::BasicBlock *BB) : ExprNode(NK_Atomic), ConditionBB(BB) {}

  static bool classof(const ExprNode *E) {
    return E->getKind() == NK_Atomic;
  }
};

class NotNode : public ExprNode {
private:
  ExprNode *Child;

public:
  NotNode(ExprNode *N) : ExprNode(NK_Not), Child(N) {}

  static bool classof(const ExprNode *E) {
    return E->getKind() == NK_Not;
  }

  ExprNode *getInternalNode() {
    return Child;
  }

};

class AndNode : public ExprNode {
private:
  ExprNode *LeftChild;
  ExprNode *RightChild;

public:
  AndNode(ExprNode *Left, ExprNode *Right) : ExprNode(NK_And), LeftChild(Left), RightChild(Right) {}

  static bool classof(const ExprNode *E) {
    return E->getKind() == NK_And;
  }

  std::pair<ExprNode *, ExprNode *> getInternalNodes() {
    return std::make_pair(LeftChild, RightChild);
  }
};

class OrNode : public ExprNode {
private:
  ExprNode *LeftChild;
  ExprNode *RightChild;

public:
  OrNode(ExprNode *Left, ExprNode *Right) : ExprNode(NK_Or), LeftChild(Left), RightChild(Right) {}

  static bool classof(const ExprNode *E) {
    return E->getKind() == NK_Or;
  }

  std::pair<ExprNode *, ExprNode *> getInternalNodes() {
    return std::make_pair(LeftChild, RightChild);
  }
};

#endif // define REVNGC_RESTRUCTURE_CFG_EXPRNODE_H
