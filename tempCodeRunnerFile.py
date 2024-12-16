 # # Step manually and plot
    # for i in tqdm(range(300)):
    #     sim.step()
    #     if i % 30 == 0:  # 5 seconds interval (5 / 0.01 = 500 steps)
    #         subplot_index = i // 50
    #         if subplot_index < num_subplots:
    #             sim.plot(ax=axes[subplot_index])