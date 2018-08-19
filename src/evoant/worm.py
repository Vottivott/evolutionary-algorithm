from circular import Circular
from ball import Ball
from evoant.fish import Fish
from muscle import Muscle
import numpy as np
from evomath import *
from itertools import izip



class Worm:
    def __init__(self, positions, ball_radius, segment_size, num_segments, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass, spring_constant, football_initial_position, football_initial_y_velocity):
        # self.num_balls = (num_segments + 1) / 2
        self.team_size = (num_segments + 1) / 2
        self.left_fish = [Fish(positions[i], ball_radius, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass) for i in range(self.team_size)]
        self.right_fish = [Fish(positions[i + self.team_size], ball_radius, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass, mirrored=True) for i in range(self.team_size)]
        for f in self.left_fish:
            f.energy = 1.0 - float(0.2*np.random.rand())
            f.age = 0.0
        for f in self.right_fish:
            f.energy = float(0.2*np.random.rand())
            f.age = 0.0
        self.football = Ball(np.array(football_initial_position), ball_radius, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass)
        self.football.velocity[1] = np.array(football_initial_y_velocity)
        # self.fish = [Fish(position + np.array([[i*segment_size],[0]]), ball_radius, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass) for i in range(self.num_balls)]
        self.fish = self.left_fish + self.right_fish
        self.balls = self.left_fish + self.right_fish + [self.football]#[f for f in self.fish]
        # self.muscles = [Muscle(b1, b2, segment_size, spring_constant) for b1,b2 in izip(self.balls[:-1],self.balls[1:])]
        self.muscles = []
        # self.balls[0].velocity[1] = -50.0
        # self.balls[1].reaching = 1.0
        self.max_y_velocity = 30.0  # 50.0
        self.max_x_velocity = 30.0  # 50.0
        self.football_max_y_velocity = 30.0  # 50.0
        self.football_max_x_velocity = 30.0  # 50.0
        self.max_shoot_velocity = 30.0
        self.max_real_muscle_length = 40.0
        self.spring_constant = spring_constant
        self.muscle_flex_length = 13.0
        self.muscle_extend_length = 28.0
        self.muscle_break_length = 40.0
        self.initial_rightmost_x = np.copy(max(b.position[0] for b in self.balls))

    def get_distance_travelled(self):
        return max(b.position[0] for b in self.balls) - self.initial_rightmost_x

    def get_x(self):
        return self.balls[0].get_x()

    def get_y(self):
        return self.balls[0].get_y()

    def step(self, level, gravity, delta_time):

        # for b in self.balls:
        #     if abs(b.velocity[0]) > self.max_x_velocity:
        #         self.max_x_velocity = max(self.max_x_velocity, abs(b.velocity[0]))
        #         print self.max_x_velocity, self.max_y_velocity
        #     if abs(b.velocity[1]) > self.max_y_velocity:
        #         self.max_y_velocity = max(self.max_y_velocity, abs(b.velocity[1]))
        #         print self.max_x_velocity, self.max_y_velocity

        #TEST
        # if self.balls[0].bounced:
        #     self.muscles[0].target_length=50.0

        for b in self.balls:
            b.planned_offset = np.array([[0.0], [0.0]])

        for f in self.fish:
            f.step(delta_time)




        for i in range(len(self.balls)):
            b = self.balls[i]
            if not b.gripping:
                b.bounce_on_level(level)
            if b.grippingness == 0.0:
                b.gripping = False

        ball_competing_shoot_velocities = []
        for i in range(len(self.balls)):
            b = self.balls[i]
            for j in range(i+1,len(self.balls)):
                other = self.balls[j]
                distSq = np.dot((other.get_position()-b.get_position()).T, other.get_position()-b.get_position())
                if distSq < (other.radius+b.radius)*(other.radius+b.radius):
                    # If collision
                    collisionLine = normalized(other.position - b.position)
                    dist = distSq**0.5
                    margin = b.radius + other.radius - dist

                    # Move to contact point (moving both balls the same distance)
                    if not b.gripping:
                        b.position += collisionLine * -margin / 2.0
                    if not other.gripping:
                        other.position += collisionLine * margin / 2.0

                    if b.reaching >= 0.5 and other.reaching >= 0.5 and other not in b.connections and b not in other.connections:
                        self.muscles.append(Muscle(b, other, b.radius + other.radius + 5.0, self.spring_constant, self.muscle_break_length))
                        b.connections.append(other)
                        other.connections.append(b)

                    u1Vector = b.velocity.T.dot(collisionLine) * collisionLine
                    u2Vector = other.velocity.T.dot(collisionLine) * collisionLine
                    u1 = collisionLine.T.dot(u1Vector)
                    u2 = collisionLine.T.dot(u2Vector)
                    m1 = b.mass
                    m2 = other.mass
                    I = m1*u1 + m2*u2
                    R = -(u2 - u1)
                    v1 = (I - m2*R) / (m1 + m2)
                    v2 = R + v1

                    v1 *= b.ball_ball_restitution
                    v2 *= other.ball_ball_restitution



                    if j == len(self.balls)-1 and b.do_shoot: # meaning other is the ball, and b wants to shoot
                        ball_competing_shoot_velocities.append(np.array(b.shoot_velocity))
                        b.start_shoot_animation = True
                    else:
                        b.velocity += -u1Vector + collisionLine * v1
                        other.velocity += -u2Vector + collisionLine * v2

                    # if i == 0: #TEST
                    #     other.velocity[0] = 30.0
                    #     other.velocity[1] = 9.0
                    # else:
                    #     b.velocity += -u1Vector + collisionLine * v1
                    #     other.velocity += -u2Vector + collisionLine * v2

            for stone in level.stones:
                distSq = np.dot((stone.get_position() - b.get_position()).T,
                                stone.get_position() - b.get_position())
                if distSq < (stone.radius + b.radius) * (stone.radius + b.radius):
                    # If collision
                    collisionLine = normalized(stone.position - b.position)
                    dist = distSq ** 0.5
                    margin = b.radius + stone.radius - dist

                    # Move to contact point (moving both balls the same distance)
                    b.position += collisionLine * -margin / 2.0


                    u1Vector = b.velocity.T.dot(collisionLine) * collisionLine
                    u2Vector = stone.velocity.T.dot(collisionLine) * collisionLine
                    u1 = collisionLine.T.dot(u1Vector)
                    u2 = collisionLine.T.dot(u2Vector)
                    m1 = b.mass
                    m2 = stone.mass
                    I = m1 * u1 + m2 * u2
                    R = -(u2 - u1)
                    v1 = (I - m2 * R) / (m1 + m2)
                    v2 = R + v1

                    v1 *= b.ball_ball_restitution
                    v2 *= stone.ball_ball_restitution

                    b.velocity += -u1Vector + collisionLine * v1

                    if stone.strength > 0.0:
                        stone.strength -= 0.04 * u1*u1
                        print stone.strength + 0.04 * u1*u1, stone.strength
                    if stone.strength <= 0.0:
                        print stone.num_foods_inside

                                            # print "bounce, %d, %d" % (i,j)
            import numpy.random
            # acceleration = numpy.random.permutation(np.array([[-1.0], [0.0]]))
            # acceleration = numpy.random.rand(1.0)*np.array([[1.0],[0.0]])

        if len(ball_competing_shoot_velocities) > 0:
            self.football.velocity = mean(ball_competing_shoot_velocities)

        # Release unreaching muscles
        i = 0
        while i < len(self.muscles):
            self.muscles[i].step(delta_time)
            if self.muscles[i].b1.reaching < 0.5 or self.muscles[i].b2.reaching < 0.5:# or np.linalg.norm(self.muscles[i].line_segment.delta) > self.muscles[i].break_length:
                self.muscles[i].b1.connections.remove(self.muscles[i].b2)
                self.muscles[i].b2.connections.remove(self.muscles[i].b1)
                del self.muscles[i]
            else:
                i += 1

        # for muscle in self.muscles:
        #     for b in self.balls:
        #         if b is not muscle.b1 and b is not muscle.b2:
        #             muscle.collide_with_ball(b)

        for i,b in enumerate(self.balls):
            # if i>0:
            if not b.gripping:
                # acceleration = gravity
                # b.velocity += acceleration * delta_time
                b.velocity *= 0.97 # viscosity

                b.position += b.velocity * delta_time
                b.position += b.planned_offset



        return True







