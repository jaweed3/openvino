// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "fold_activation_zero_point.hpp"

#include <memory>
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/core/rt_info.hpp"

using namespace ov::pass;

ov::intel_cpu::FoldActivationZeroPoint::FoldActivationZeroPoint() {
    // pattern definition
    // callback
    // register matcher
    auto zp_const = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto zp_convert = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({zp_const});
    auto zp_input = std::make_shared<ov::pass::pattern::op::Or>(
        ov::OutputVector{zp_const, zp_convert});
    auto activation = ov::pass::pattern::any_input();
    auto subtract = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({activation, zp_input});
    auto conv = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>(
        {subtract, ov::pass::pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto conv_node = ov::as_type_ptr<ov::op::v1::Convolution>(
            pattern_map.at(conv).get_node_shared_ptr());
        auto subtract_node = pattern_map.at(subtract).get_node_shared_ptr();
        auto zp_const_node = ov::as_type_ptr<ov::op::v0::Constant>(
            pattern_map.at(zp_const).get_node_shared_ptr());

        if (!conv_node || !subtract_node || !zp_const_node) {
            return false;
        }

        auto zp_values = zp_const_node->cast_vector<int32_t>();
        if (zp_values.empty()) return false;

        conv_node->get_rt_info()["ActivationZeroPoint"] = zp_values[0];

        conv_node->input(0).replace_source_output(subtract_node->input_value(0));

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv, "FoldActivationZeroPoint");
    this->register_matcher(m, callback);
}
