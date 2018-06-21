from time import time


def start_game(env, agent, on_policy=True):
    # Train.
    if agent.mode == 'train':
        for episode in range(agent.train_episodes):
            s, r_episode, now = env.reset(), 0, time()
            while True:
                a = agent.predict(s)
                s_n, r, done, _ = env.step(a)
                r_episode += r
                if on_policy:
                    agent.snapshot(s, a, r, s_n)
                else:
                    agent.snapshot(s, a, r_episode, s_n)
                s = s_n
                if done:
                    # Logs.
                    agent.logger.warning('Episode: {} | Times: {} | Rewards: {}'.format(episode,
                                                                                        time() - now,
                                                                                        r_episode))
                    break
            agent.train()
            if episode % 50 == 0:
                agent.save()
    elif agent.mode == 'test':
        agent.restore()
        # Reset env.
        s, r_episode, now = env.reset(), 0, time()
        while True:
            a = agent.predict(s)
            s_n, r, done, _, = env.step(a)
            r_episode += r
            s = s_n
            if done:
                agent.logger.warning('Test mode, rewards: {}'.format(r_episode))
                break
