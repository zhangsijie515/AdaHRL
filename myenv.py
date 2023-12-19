import numpy as np
from minigrid.minigrid_env import Grid, MiniGridEnv, Key, Door, Goal, MissionSpace, Window


class MyEnv(MiniGridEnv):
    def __init__(self, width=11, height=11, max_steps=None):
        self.R = 0
        self.step_count = 0
        self.carrying = None

        if max_steps is None:
            max_steps = width * height
        mission_space = MissionSpace(
            mission_func=lambda: "use the keys to open the doors and then get to the goal"
        )
        super().__init__(
            mission_space=mission_space,
            max_steps=max_steps,
            width=width,
            height=height
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # 放置墙体
        self.grid.wall_rect(0, 0, width, height)
        self.grid.wall_rect(1, 1, 2, 2)
        self.grid.wall_rect(8, 1, 2, 2)
        self.grid.wall_rect(1, 8, 2, 2)
        self.grid.wall_rect(8, 8, 2, 2)
        self.grid.horz_wall(1, 3, width - 2)
        self.grid.horz_wall(1, 7, width - 2)
        self.grid.vert_wall(3, 1, height - 2)
        self.grid.vert_wall(7, 1, height - 2)

        # 放置门
        self.put_obj(Door('red', is_locked=True), 5, 3)
        self.put_obj(Door('blue', is_locked=True), 7, 5)
        self.put_obj(Door('green', is_locked=True), 5, 7)
        self.put_obj(Door('yellow', is_locked=True), 3, 5)

        # 放置钥匙
        self.put_obj(Key('red'), 4, 4)
        self.put_obj(Key('green'), 5, 1)
        self.put_obj(Key('blue'), 5, height - 2)
        self.put_obj(Key('yellow'), width - 2, 5)

        # 放置最终目标
        self.put_obj(Goal(), 1, 5)

        # 放置智能体
        self.agent_pos = (5, 5)
        self.agent_dir = 0

    def reset(self, *, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = []

        # Step count adn R since episode start
        self.step_count = 0
        self.R = 0

        # Return first observation
        obs = self.agent_pos

        if not return_info:
            return obs
        else:
            return obs, {}

    def render(self, mode="human", highlight=True, tile_size=32):
        assert mode in self.metadata["render_modes"]
        """
        Render the whole-grid human view
        """
        if mode == "human" and not self.window:
            self.window = Window("gym_minigrid")
            self.window.show(block=False)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
        )

        if mode == "human":
            self.window.set_caption(self.mission)
            self.window.show_img(img)
        else:
            return img

    # meta
    # 拿到钥匙: r = 10
    # 到达终点: r = 100
    # else:   r = 0

    # controller
    # 时间惩罚: r = -1
    # 到达子目标: r = 10(在train函数中修改)

    def step(self, action):
        reward = 0  # 环境本身的奖励（meta_controller使用）
        reward_ = -1  # 内在奖励（controller使用）
        done = False
        keys = []
        win = False

        # up
        if action == 0:
            next_agent_pos = self.agent_pos + np.array((0, -1))
        # down
        if action == 1:
            next_agent_pos = self.agent_pos + np.array((0, +1))
        # right
        if action == 2:
            next_agent_pos = self.agent_pos + np.array((1, 0))
        # left
        if action == 3:
            next_agent_pos = self.agent_pos + np.array((-1, 0))

        next_cell = self.grid.get(*next_agent_pos)
        # 空地
        if next_cell is None:
            self.agent_pos = next_agent_pos
        # 非空地：包括可重叠和不可重叠
        else:
            # 若重叠
            if next_cell.can_overlap():
                # 已经打开的门
                if next_cell.type == 'door':
                    self.agent_pos = next_agent_pos
                # 最终目标，游戏结束
                if next_cell.type == 'goal':
                    done = True
                    reward = 100
                    win = True
                self.agent_pos = next_agent_pos
            # 不可重叠----钥匙
            elif next_cell is not None and next_cell.type == 'key':
                reward = 10
                self.carrying.append(next_cell)
                self.grid.set(*next_agent_pos, None)
                self.agent_pos = next_agent_pos
            # 不可重叠----关闭的门
            elif next_cell is not None and next_cell.type == 'door':
                for i in self.carrying:
                    if i.color == next_cell.color:
                        next_cell.is_open = True
                        self.carrying.pop(self.carrying.index(i))
                        self.agent_pos = next_agent_pos

        self.step_count += 1
        self.R += reward
        if self.step_count >= self.max_steps:
            done = True

        # 返回手上还剩下的钥匙
        if len(self.carrying) > 0:
            for i in self.carrying:
                keys.append(i.color)

        obs = (self.agent_pos-1).tolist()
        return obs, reward, reward_, done, {'steps': self.step_count, 'keys': keys, 'win': win, 'R': self.R}
