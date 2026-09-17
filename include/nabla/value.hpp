
#pragma once
#include <functional>
#include <memory>
#include <vector>

namespace nabla {

struct Node {
    float data = 0.0f;
    float grad = 0.0f;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void(float upstream)> backward_fn;

    explicit Node(float d) : data(d) {}
};

class Value {

  private:
    explicit Value(std::shared_ptr<Node> n) : node_(std::move(n)) {}
    std::shared_ptr<Node> node_;

  public:
    explicit Value(float d) : node_(std::make_shared<Node>(d)) {}

    float data() const { return node_->data; }
    float grad() const { return node_->grad; }

    friend Value operator+(const Value &a, const Value &b);
    friend Value operator*(const Value &a, const Value &b);

    void backward();
};

//
// Now 01:41 I don't know wtf that i'm doing rn. - Alex
//

inline Value operator+(const Value &a, const Value &b) {
    auto out = std::make_shared<Node>(a.node_->data + b.node_->data);
    auto a_node = a.node_;
    auto b_node = b.node_;
    out->parents = {a_node, b_node};
    out->backward_fn = [a_node, b_node](float upstream) {
        a_node->grad += upstream;
        b_node->grad += upstream;
    };
    return Value(out);
}

inline Value operator*(const Value &a, const Value &b) {
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
    return Value(out);
}

} // namespace nabla
