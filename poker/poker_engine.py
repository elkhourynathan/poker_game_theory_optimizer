import numpy as np
from treys import Deck, Evaluator, Card
from typing import List, Tuple

class PokerEngine:
    def __init__(self, num_players=2, player_hand=[], community_cards=[], decision_threshold=0.5):
        self.eval = Evaluator()
        self.player_hand = player_hand
        self.community_cards = community_cards
        self.num_players = num_players
        self.decision_threshold = decision_threshold
    
    def set_player_hand(self, player_hand: List[str]) -> None:
        self.player_hand = [CardFactory.create_card(card_str) for card_str in player_hand]

    def set_community_cards(self, community_cards: List[str]) -> None:
        self.community_cards = [CardFactory.create_card(card_str) for card_str in community_cards]

    def set_num_players(self, num_players: int) -> None:
        self.num_players = num_players
    
    def monte_carlo_simulation(self, num_simulations: int = 10000) -> Tuple[float, float, float]:
        """
        Runs a monte carlo simulation defaulted a 10000 times to determine the probability of winning, tying, and losing a hand.
        Based on the current available community cards seen, the current player hand, and the number of players in the game.
        Args:
            num_simulations (int): Number of simulations to run
        Returns:
            win_prob (float): Probability of winning the hand
            tie_prob (float): Probability of tying the hand
            loss_prob (float): Probability of losing the hand
        """
        wins = 0
        ties = 0
        losses = 0

        for _ in range(num_simulations):
            deck = Deck()
            deck.shuffle()

            # "Draw" the community cards and player cards from deck
            
            for card in self.player_hand + self.community_cards:
                deck.cards.remove(card)
            
            # Draw cards for the other players
            other_players_hands = [deck.draw(2) for _ in range(self.num_players - 1)]

            # Draw remaining community cards if necessary
            remaining_community_cards = self.community_cards + deck.draw(5 - len(self.community_cards))

            # Evaluate the hands

            player_hand_strength = self.eval.evaluate(remaining_community_cards, self.player_hand)
            other_players_hand_strengths = [self.eval.evaluate(remaining_community_cards, hand) for hand in other_players_hands]

            # Calculate win, tie, loss
            best_other_player_strength = min(other_players_hand_strengths)

            if player_hand_strength < best_other_player_strength:
                wins += 1
            elif player_hand_strength == best_other_player_strength:
                ties += 1
            else:
                losses += 1
        
        win_prob = wins / num_simulations
        tie_prob = ties / num_simulations
        loss_prob = losses / num_simulations

        return win_prob, tie_prob, loss_prob

    def decide_action(self, win_prob, tie_prob, loss_prob, pot_size, current_bet):
        """
        Decide whether to fold, call, or raise based on probabilities.
        Assumes that a tie results in the pot being split.

        Args:
            win_prob (float): Probability of winning the hand
            tie_prob (float): Probability of tying the hand
            loss_prob (float): Probability of losing the hand
            pot_size (float): Current size of the pot
            current_bet (float): Current bet size
        Returns:
            Recommened action (str): 'fold', 'call', or 'raise'
        """

        # Calculate expected value
        ev = (pot_size * win_prob) + ((pot_size / 2) * tie_prob) - (current_bet * loss_prob)

        # Decide action based on expected value
        if ev > current_bet * self.decision_threshold:
            return 'raise'
        elif ev > -current_bet * self.decision_threshold:
            return 'call'
        else:
            return 'fold'


class CardFactory:
    @staticmethod
    def create_card(card_str: str) -> Card:
        """
        Creates a treys.Card object from a string representation of a card.
        Args:
            card_str (str): String representation of a card
        Returns:
            Card: treys.Card object
        """
        # Replace "10" with "T" at the start of the string
        if card_str.startswith("10"):
            card_str = "T" + card_str[2:]

        # Lower case the second letter to match treys.Card format
        if len(card_str) != 2:
            print("Invalid card format should be 2 characters long. e.g. 'Ah' for Ace of Hearts.")
            return None

        card_str = card_str[:-1] + card_str[-1].lower()
        return Card.new(card_str)