#pragma once
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

namespace nabla {

struct Node {
    float data = 0.0f;
    float grad = 0.0f;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void(float upstream)> backward_fn;

    explicit Node(float d) : data(d) {}
};

class Scalar {

  private:
    explicit Scalar(std::shared_ptr<Node> n) : node_(std::move(n)) {}
    std::shared_ptr<Node> node_;

  public:
    Scalar(float d) : node_(std::make_shared<Node>(d)) {}

    float data() const { return node_->data; }
    float grad() const { return node_->grad; }

    friend Scalar operator+(const Scalar &a, const Scalar &b);
    friend Scalar operator*(const Scalar &a, const Scalar &b);
    friend Scalar operator-(const Scalar &a, const Scalar &b);

    void backward() const;
    void step(float lr) { node_->data -= lr * node_->grad; }
};

//
// Now 01:41 I don't know wtf that i'm doing rn. - Alex
//

inline Scalar operator+(const Scalar &a, const Scalar &b) {
    auto out = std::make_shared<Node>(a.node_->data + b.node_->data);
    auto a_node = a.node_;
    auto b_node = b.node_;
    out->parents = {a_node, b_node};
    out->backward_fn = [a_node, b_node](float upstream) {
        a_node->grad += upstream;
        b_node->grad += upstream;
    };
    return Scalar(out);
}

inline Scalar operator-(const Scalar &a, const Scalar &b) {
    auto out = std::make_shared<Node>(a.node_->data - b.node_->data);
    auto a_node = a.node_;
    auto b_node = b.node_;
    out->parents = {a_node, b_node};
    out->backward_fn = [a_node, b_node](float upstream) {
        a_node->grad += upstream;
        b_node->grad -= upstream;
    };
    return Scalar(out);
}

inline Scalar operator*(const Scalar &a, const Scalar &b) {
    auto out = std::make_shared<Node>(a.node_->data * b.node_->data);
    auto a_node = a.node_;
    auto b_node = b.node_;
    out->parents = {a_node, b_node};
    out->backward_fn = [a_node, b_node](float upstream) {
        const float da = upstream * b_node->data;
        const float db = upstream * a_node->data;

        a_node->grad += da;
        b_node->grad += db;
    };
    return Scalar(out);
}

// AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH

inline void Scalar::backward() const {
    std::vector<Node *> order;
    std::unordered_set<const Node *> visited;
    std::vector<std::pair<Node *, std::size_t>> stack;

    stack.emplace_back(node_.get(), 0);
    visited.insert(node_.get());
    while (!stack.empty()) {
        auto &[n, i] = stack.back();
        if (i < n->parents.size()) {
            Node *next = n->parents[i++].get();
            if (visited.insert(next).second) {
                stack.emplace_back(next, 0);
            }
        } else {
            order.push_back(n);
            stack.pop_back();
        }
    }
    for (Node *n : order) {
        n->grad = 0.0f;
    }
    node_->grad = 1.0f;

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn((*it)->grad);
        }
    }
}

} // namespace nabla
