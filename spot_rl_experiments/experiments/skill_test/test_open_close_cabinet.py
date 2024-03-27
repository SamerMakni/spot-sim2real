# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    from perception_and_utils.utils.generic_utils import map_user_input_to_boolean

    # Init the skill
    spotskillmanager = SpotSkillManager(use_mobile_pick=True)

    # Using while loop
    contnue = True
    while contnue:
        spotskillmanager.opencabinet()
        spotskillmanager.pick("bottle")
        breakpoint()
        close_drawer = map_user_input_to_boolean(
            "Do you want to close the cabinet ? Y/N "
        )
        if close_drawer:
            spotskillmanager.closecabinet()
        contnue = map_user_input_to_boolean(
            "Do you want to open the cabinet again ? Y/N "
        )
