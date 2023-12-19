import random
from tool import *
import numpy as np
from Model import tf


class MetaControllerOurs:
    def __init__(self, env, controller, meta_controller, max_episodes, max_meta_steps, controller_buffer_num,
                 meta_controller_buffer_num, knowledge_epsilon, epsilon
                 , gamma, lr, update_controller_batch, update_meta_batch, retain_function):
        self.current_goal = None  
        self.goal_from_meta = None  
        self.goals = None 
        self.s = None  
        self.done = None
        self.destination = None
        self.meta_track_buffer = MetaTrackBuffer(100)  
        self.env = env
        self.controller = controller
        self.meta_controller = meta_controller
        self.controller_buffer = ControllerBuffer(num=controller_buffer_num)
        self.meta_controller_buffer = MetaControllerBuffer(num=meta_controller_buffer_num)
        self.max_episode = max_episodes
        self.max_meta_steps = max_meta_steps  
        self.knowledge_epsilon = knowledge_epsilon
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.fuc = retain_function
        self.update_controller_batch = update_controller_batch
        self.update_meta_batch = update_meta_batch
        self.meta_controller.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr),
                                     loss=tf.keras.losses.MeanSquaredError())
        self.controller.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss=tf.keras.losses.MeanSquaredError())

    def return_meta_goal(self):
        next_goal = None
        for i in self.current_goal.same_level:
            goal = self.to_goal_class(i)
            if not goal.arrived:
                next_goal = goal
                break
        if not next_goal:
            for i in self.current_goal.father:
                goal = self.to_goal_class(i)
                if not goal.arrived:
                    next_goal = goal
                    break
        if not next_goal:
            if not self.destination:
                return None
            else:
                return self.destination
        return next_goal.get_value()
    def to_goal_class(self, s):
        return label_to_goal(s, self.goals)

    def to_current_goal_index(self):
        return self.goals.index(self.current_goal)

    def arrive_s(self, s):
        for goal in self.goals:
            if goal.get_value() == s:
                goal.has_arrived()
                self.current_goal = goal

    def init_arrive(self):
        for goal in self.goals:
            goal.init_arrived()
        self.goals[0].has_arrived()

    def update_goals(self, goals, Thr=0.25):
        for goal in goals:
            sub_goal_children, sub_goal_father = self.meta_track_buffer.find_father_and_children(goal.get_value(),
                                                                                                 goals)
            sub_goal_father = return_accept_goal(sub_goal_father, Thr)
            sub_goal_children = return_accept_goal(sub_goal_children, Thr)
            for i in sub_goal_father:
                if i.get_value() in goal.children:
                    goal.delete_children(i.get_value())
                    goal.delete_father(i.get_value())
                    goal.add_same_level(i.get_value())
                elif i.get_value() in goal.same_level:
                    pass
                else:
                    goal.add_father(i.get_value())
            for i in sub_goal_children:
                if i.get_value() in goal.father:
                    goal.delete_children(i.get_value())
                    goal.delete_father(i.get_value())
                    goal.add_same_level(i.get_value())
                elif i.get_value() in goal.same_level:
                    pass
                else:
                    goal.add_children(i.get_value())

        return goals

    def to_meta_controller_input(self):
        Input = np.zeros((1, 6))
        Input[0][self.to_current_goal_index()] = 1
        return Input

    def choose_meta_goal(self, epsilon, knowledge_epsilon):
        if random.random() < epsilon:
            if random.random() < knowledge_epsilon:
                goals_value = self.meta_controller.predict(self.to_meta_controller_input())
                for i in range(len(goals_value[0])):
                    if i >= len(self.goals):
                        goals_value[0][i] = -np.inf
                goal_index = np.argmax(goals_value)
                return self.goals[goal_index].get_value()
            else:
                self.add_knowledge_epsilon()
                return self.return_meta_goal()
        else:
            return self.goals[random.randint(1, len(self.goals) - 1)].get_value()

    def to_controller_input(self):
        s = np.array(self.s)
        goal_from_meta = np.array(self.goal_from_meta)
        Input = np.concatenate((s, goal_from_meta)).reshape(-1, 2, 2) 
        Input = tf.one_hot(Input, 9, axis=-1, dtype=tf.float32)
        return Input

    def choose_controller_action(self, epsilon):
        if random.random() < epsilon:

            action_value = self.controller.predict(self.to_controller_input())
            action = np.argmax(action_value)
            return action
        else:
            return random.randint(0, 3)

    def update_meta_controller(self, batch_size=32):
        if self.meta_controller_buffer.__len__() >= batch_size:
            current_goal, arrived_goal, r = self.meta_controller_buffer.sample(batch_size)
            current_goal = tf.one_hot(current_goal, 6)
            arrived_goal = tf.one_hot(arrived_goal, 6)
            r = tf.convert_to_tensor(r, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(self.meta_controller.variables)
                Q = tf.reduce_sum(self.meta_controller(current_goal) * arrived_goal, axis=1)
                Q_target = tf.reduce_max(self.meta_controller(arrived_goal), axis=1)
                loss = tf.reduce_mean(tf.square(self.gamma * Q_target + r - Q))
            grad = tape.gradient(loss, self.meta_controller.variables)
            self.meta_controller.optimizer.apply_gradients(zip(grad, self.meta_controller.variables))
            return loss
        else:
            return np.inf

    def update_controller(self, batch_size=64):
        if self.controller_buffer.__len__() >= batch_size:
            s, goal, a, r_, s_ = self.controller_buffer.sample(batch_size)
            Q_input = np.concatenate((s, goal), axis=1).reshape((-1, 2, 2))
            Q_input = tf.one_hot(Q_input, 9, axis=-1, dtype=tf.float32)
            Q_target_input = np.concatenate((s_, goal), axis=1).reshape((-1, 2, 2))
            Q_target_input = tf.one_hot(Q_target_input, 9, axis=-1, dtype=tf.float32)
            r_ = tf.convert_to_tensor(r_, dtype=tf.float32)
            a = tf.one_hot(a, 4)
            with tf.GradientTape() as tape:
                tape.watch(self.controller.variables)
                Q = tf.reduce_sum(a * self.controller(Q_input), axis=1)
                Q_target = tf.reduce_max(self.controller(Q_target_input), axis=1)
                loss = tf.reduce_mean(tf.square(self.gamma * Q_target + r_ - Q))
            grad = tape.gradient(loss, self.controller.variables)
            self.controller.optimizer.apply_gradients(zip(grad, self.controller.variables))
            return loss
        else:
            return np.inf

    def train(self):
        np.set_printoptions(precision=3)
        f = open(self.fuc + "_find_goal_ours.txt", 'w')
        f.close()
        self.goals = random_walk(env=self.env, max_episodes=10, file=self.fuc + "_find_goal_ours.txt")
        epsilon = [self.epsilon] * len(self.goals)
        retain_rate = [1] * len(self.goals)
        R = []
        step = []
        for i in range(self.max_episode):
            self.done = False
            self.s = np.array(self.env.reset())
            self.goal_from_meta = None
            self.current_goal = self.goals[0]
            every_goal_steps = 0
            track = [0]
            last_s = self.s
            flag = False 
            self.init_arrive() 
            while not self.done:
                if not self.goal_from_meta and not flag:
                    if not self.destination:
                        self.goal_from_meta = self.return_meta_goal()
                    else:
                        self.goal_from_meta = self.choose_meta_goal(epsilon=0.9,
                                                                    knowledge_epsilon=self.knowledge_epsilon)
                    flag = True

                if not self.goal_from_meta:
                    a = random.randint(0, 3)
                else:
                    index = self.goals.index(self.to_goal_class(self.goal_from_meta))
                    a = self.choose_controller_action(epsilon=epsilon[index])
                self.s, r, r_, self.done, info = self.env.step(a)
                every_goal_steps += 1
                if r > 0:
                    if in_sub_goals(self.s, self.goals):
                        track.append(self.s)
                        index = self.goals.index(self.to_goal_class(self.s))
                        if random.random() < retain_rate[index]:
                            self.meta_controller_buffer.push(self.to_current_goal_index(), index, r)
                        self.arrive_s(self.s) 

                    else:
                        if info['win']:
                            f = open(self.fuc + '_find_goal_ours.txt', 'a')
                            print('{}: done{}'.format((i + 1), self.s), file=f)
                            self.destination = self.s
                            f.close()
                        else:
                            f = open(self.fuc + '_find_goal_ours.txt', 'a')
                            print('{}: new subgoal {}'.format((i + 1), self.s), file=f)
                            f.close()
                        self.goals.append(SubGoal(self.s))
                        epsilon.append(self.epsilon)
                        retain_rate.append(1)
                        track.append(self.s)

                        self.meta_controller_buffer.push(self.to_current_goal_index(),
                                                         self.goals.index(self.to_goal_class(self.s)), r)
                        self.arrive_s(self.s) 

                if self.goal_from_meta:
                    if self.s == self.goal_from_meta:
                        r_ = 10
                        self.controller_buffer.push(last_s, self.goal_from_meta, a, r_, self.s)
                        self.goal_from_meta = None 
                        flag = False
                        every_goal_steps = 0
                    else:
                        index = self.goals.index(self.to_goal_class(self.goal_from_meta))
                        if random.random() < retain_rate[index]:  
                            self.controller_buffer.push(last_s, self.goal_from_meta, a, r_, self.s)

                if in_sub_goals(self.s, self.goals):
                    index = self.goals.index(self.to_goal_class(self.s))
                    if random.random() < retain_rate[index]:  
                        self.controller_buffer.push(last_s, self.s, a, 10, self.s)

                if every_goal_steps >= self.max_meta_steps and flag:
                    self.goal_from_meta = None  
                    flag = False
                    every_goal_steps = 0

                last_s = self.s

            epsilon = self.add_epsilon(track, epsilon)
            R.append(info['R'])
            step.append(info['steps'])
            controller_loss = self.update_controller(batch_size=self.update_controller_batch)
            meta_controller_loss = self.update_meta_controller(batch_size=self.update_meta_batch)
            self.meta_track_buffer.push(track)
            if (i + 1) % 20 == 0:
                self.goals = self.update_goals(self.goals)

            if (i + 1) % 1000 == 0:
                f = open(self.fuc + '_R_ours.txt', 'w')
                for line in R:
                    f.write(str(line) + '\n')
                f.close()
                f = open(self.fuc + '_step_ours.txt', 'w')
                for line in step:
                    f.write(str(line) + '\n')
                f.close()

            if (i + 1) % 1000 == 0:
                epsilon, retain_rate, self.knowledge_epsilon = self.meta_track_buffer.calculate_retain_rate_and_epsilon(
                    self.goals, fuc=self.fuc)
            if (i + 1) % 100 == 0:
                f = open(self.fuc + '_controller_buffer_rate.txt', 'a')
                f.write(str(self.controller_buffer.calculate_buffer_date_rate(self.goals)) + '\n')
                f.close()

            print('episode:{}   C_loss:{:.2f}   M_loss:{:.2f}   e:{}   retain:{}   k_e:{:.2f}   R:{}   track:{}'.format(
                i + 1,
                controller_loss,
                meta_controller_loss,
                np.array(epsilon),
                np.array(retain_rate),
                self.knowledge_epsilon,
                info['R'],
                track))
