from poker import PokerEngine
from treys import Card


def main():
    # Initialize the PokerEngine
    poker_engine = PokerEngine()
    
    # Set the player hand
    poker_engine.set_player_hand([Card.new('As'), Card.new('Ks')])
    
    # Set the community cards
    poker_engine.set_community_cards([Card.new('Ah'), Card.new('Kd'), Card.new('Qh')])
    
    # Set the number of players
    poker_engine.set_num_players(3)
    
    # Run the monte carlo simulation
    win_prob, tie_prob, loss_prob = poker_engine.monte_carlo_simulation()
    
    print(f"Win Probability: {win_prob*100:.2f}%")
    print(f"Tie Probability: {tie_prob*100:.2f}%")
    print(f"Loss Probability: {loss_prob*100:.2f}%")

    # Decision
    decision = poker_engine.decide_action(win_prob, tie_prob, loss_prob, 50, 100)
    print(decision)


if __name__ == "__main__":
    main()