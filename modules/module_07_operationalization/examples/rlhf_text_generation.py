#!/usr/bin/env python3
"""
RLHF (Reinforcement Learning from Human Feedback) for Text Generation.

This example demonstrates the core concepts of RLHF used to train language models
like ChatGPT. It uses a simplified setup with character-level generation and
simple tasks to illustrate the key components.

RLHF Pipeline:
1. Supervised Fine-Tuning (SFT): Train on demonstrations
2. Reward Model Training: Learn from human preferences
3. PPO Optimization: Optimize policy against reward model

This is an educational example. Production RLHF uses:
- Large transformer models (GPT, LLaMA)
- Subword tokenization (BPE, SentencePiece)
- Distributed training
- KL penalty to prevent reward hacking

Example:
  # Train RLHF pipeline on simple text task
  python rlhf_text_generation.py --task sentiment --iterations 100

  # With different tasks
  python rlhf_text_generation.py --task length --iterations 100
  python rlhf_text_generation.py --task caps --iterations 100

Reference:
  Ouyang et al. (2022) "Training language models to follow instructions with human feedback"
  https://arxiv.org/abs/2203.02155

Note: This uses character-level models for educational clarity. Production RLHF
uses tokenizer-based transformer architectures.
"""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError as e:
    raise SystemExit("Requires: torch") from e


@dataclass
class Config:
    # Model architecture
    vocab_size: int = 128  # ASCII characters
    embed_dim: int = 64
    hidden_dim: int = 128
    max_length: int = 20

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    ppo_clip_eps: float = 0.2
    kl_coef: float = 0.1  # KL penalty coefficient
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # PPO
    ppo_epochs: int = 4
    gamma: float = 1.0  # No discounting for text generation

    # RLHF stages
    sft_iterations: int = 100
    reward_model_iterations: int = 100
    rlhf_iterations: int = 100

    seed: int = 42
    device: str = "cpu"


class PolicyNetwork(nn.Module):
    """
    Simple RNN-based language model for text generation.

    In production RLHF: This would be a transformer (GPT, LLaMA, etc.)
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: [batch, seq_len] token indices
            hidden: RNN hidden state

        Returns:
            logits: [batch, seq_len, vocab_size]
            hidden: Updated hidden state
        """
        embedded = self.embed(x)  # [batch, seq_len, embed_dim]
        out, hidden = self.rnn(embedded, hidden)  # [batch, seq_len, hidden_dim]
        logits = self.fc(out)  # [batch, seq_len, vocab_size]
        return logits, hidden

    def generate(self, prompt: str, max_length: int, temperature: float = 1.0) -> str:
        """Generate text given a prompt."""
        device = next(self.parameters()).device
        tokens = [ord(c) for c in prompt]
        original_prompt_len = len(tokens)
        
        # Handle empty prompts by adding a start token (space)
        if not tokens:
            tokens = [ord(' ')]
        hidden = None

        for _ in range(max_length - len(prompt)):
            x = torch.LongTensor(np.array([tokens])).to(device)
            with torch.no_grad():
                logits, hidden = self.forward(x, hidden)
                # Sample from last token distribution
                probs = F.softmax(logits[0, -1] / temperature, dim=0)
                next_token = torch.multinomial(probs, 1).item()

            tokens.append(next_token)

            # Stop at newline or null
            if next_token in [0, ord('\n')]:
                break

        # For empty prompts, skip the artificial start token
        start_idx = 1 if original_prompt_len == 0 else 0
        return ''.join([chr(min(max(t, 0), 127)) for t in tokens[start_idx:]])


class ValueNetwork(nn.Module):
    """Value network for PPO (critic)."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len] token indices

        Returns:
            values: [batch, seq_len] state values
        """
        embedded = self.embed(x)
        out, _ = self.rnn(embedded)
        values = self.fc(out).squeeze(-1)
        return values


class RewardModel(nn.Module):
    """
    Reward model that scores text quality.

    In production RLHF: This is trained on human preference comparisons.
    Here: We use simple heuristic rewards for demonstration.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Score a sequence.

        Args:
            x: [batch, seq_len] token indices

        Returns:
            reward: [batch] scalar reward for each sequence
        """
        embedded = self.embed(x)
        out, _ = self.rnn(embedded)
        # Use final hidden state for sequence reward
        reward = self.fc(out[:, -1]).squeeze(-1)
        return reward


def create_synthetic_demonstrations(task: str, num_samples: int = 1000) -> list[str]:
    """
    Create synthetic demonstrations for supervised fine-tuning.

    In production: These would be human-written examples.
    """
    demos = []

    if task == "sentiment":
        # Generate positive sentiment text
        positive_words = ["good", "great", "amazing", "wonderful", "excellent", "fantastic", "love"]
        for _ in range(num_samples):
            text = f"I think this is {random.choice(positive_words)}!"
            demos.append(text)

    elif task == "length":
        # Generate text of specific length
        for _ in range(num_samples):
            length = random.randint(10, 15)
            text = "x" * length
            demos.append(text)

    elif task == "caps":
        # Generate uppercase text
        words = ["HELLO", "WORLD", "TEST", "CAPS", "TEXT"]
        for _ in range(num_samples):
            text = " ".join(random.sample(words, k=random.randint(2, 4)))
            demos.append(text)

    else:
        raise ValueError(f"Unknown task: {task}")

    return demos


def create_preference_pairs(task: str, num_pairs: int = 500) -> list[tuple[str, str, int]]:
    """
    Create preference pairs for reward model training.

    Returns: List of (text1, text2, preferred_idx) where preferred_idx ∈ {0, 1}

    In production: These would come from human annotators comparing outputs.
    """
    pairs = []

    if task == "sentiment":
        positive_words = ["good", "great", "amazing", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "horrible"]

        for _ in range(num_pairs):
            pos_text = f"This is {random.choice(positive_words)}!"
            neg_text = f"This is {random.choice(negative_words)}."
            # Positive sentiment preferred
            pairs.append((neg_text, pos_text, 1))

    elif task == "length":
        for _ in range(num_pairs):
            short_text = "x" * random.randint(5, 9)
            long_text = "x" * random.randint(10, 15)
            # Longer text preferred (within reason)
            pairs.append((short_text, long_text, 1))

    elif task == "caps":
        for _ in range(num_pairs):
            lower_text = "hello world"
            upper_text = "HELLO WORLD"
            # Uppercase preferred
            pairs.append((lower_text, upper_text, 1))

    return pairs


def train_sft(policy: PolicyNetwork, demonstrations: list[str], cfg: Config):
    """
    Stage 1: Supervised Fine-Tuning on demonstrations.

    Train the policy to imitate good examples (behavioral cloning).
    """
    console.print("[bold]Stage 1: Supervised Fine-Tuning[/bold]")

    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    for iteration in range(cfg.sft_iterations):
        # Sample batch
        batch_texts = random.sample(demonstrations, min(cfg.batch_size, len(demonstrations)))

        # Convert to token sequences
        token_seqs = []
        for text in batch_texts:
            tokens = [ord(c) for c in text[:cfg.max_length]]
            tokens = tokens + [0] * (cfg.max_length - len(tokens))  # Pad
            token_seqs.append(tokens)

        x = torch.LongTensor(token_seqs).to(cfg.device)

        # Teacher forcing: predict next token
        logits, _ = policy(x[:, :-1])
        targets = x[:, 1:]

        # Cross-entropy loss
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            console.log(f"SFT Iteration {iteration}/{cfg.sft_iterations} | Loss: {loss.item():.4f}")

    console.print("[green]✓ SFT complete[/green]\n")


def train_reward_model(reward_model: RewardModel, preference_pairs: list[tuple[str, str, int]], cfg: Config):
    """
    Stage 2: Train reward model on human preferences.

    Learn to predict which text is better according to human preferences.
    """
    console.print("[bold]Stage 2: Reward Model Training[/bold]")

    optimizer = optim.Adam(reward_model.parameters(), lr=cfg.learning_rate)

    for iteration in range(cfg.reward_model_iterations):
        # Sample batch
        batch_pairs = random.sample(preference_pairs, min(cfg.batch_size, len(preference_pairs)))

        text1_list, text2_list, preferred_list = zip(*batch_pairs)

        # Tokenize
        def tokenize(texts):
            token_seqs = []
            for text in texts:
                tokens = [ord(c) for c in text[:cfg.max_length]]
                tokens = tokens + [0] * (cfg.max_length - len(tokens))
                token_seqs.append(tokens)
            return torch.LongTensor(token_seqs).to(cfg.device)

        x1 = tokenize(text1_list)
        x2 = tokenize(text2_list)

        # Get rewards
        r1 = reward_model(x1)
        r2 = reward_model(x2)

        # Preference loss (Bradley-Terry model)
        # P(text2 > text1) = σ(r2 - r1)
        preferred_t = torch.FloatTensor(preferred_list).to(cfg.device)

        # If preferred=1, we want r2 > r1, so maximize σ(r2 - r1)
        # If preferred=0, we want r1 > r2, so maximize σ(r1 - r2)
        logits = torch.where(
            preferred_t == 1,
            r2 - r1,
            r1 - r2
        )

        loss = -F.logsigmoid(logits).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            accuracy = ((logits > 0).float().mean().item())
            console.log(f"RM Iteration {iteration}/{cfg.reward_model_iterations} | Loss: {loss.item():.4f} | Acc: {accuracy:.2%}")

    console.print("[green]✓ Reward model training complete[/green]\n")


def train_rlhf_ppo(policy: PolicyNetwork, value_net: ValueNetwork, reward_model: RewardModel,
                   ref_policy: PolicyNetwork, cfg: Config):
    """
    Stage 3: RLHF with PPO.

    Optimize policy to maximize reward model scores, with KL penalty to stay close
    to reference policy (prevents reward hacking).
    """
    console.print("[bold]Stage 3: RLHF with PPO[/bold]")

    policy_optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=cfg.learning_rate)

    prompts = ["", "I", "The", "This"]  # Simple prompts

    for iteration in range(cfg.rlhf_iterations):
        # Generate rollouts
        batch_prompts = [random.choice(prompts) for _ in range(cfg.batch_size)]

        generated_texts = []
        token_seqs = []
        log_probs_list = []
        ref_log_probs_list = []

        for prompt in batch_prompts:
            tokens = [ord(c) for c in prompt]
            original_prompt_len = len(tokens)
            
            # Handle empty prompts by adding a start token (space)
            if not tokens:
                tokens = [ord(' ')]
            
            log_probs = []
            ref_log_probs = []

            # Generate sequence
            hidden = None
            ref_hidden = None

            for _ in range(cfg.max_length - len(prompt)):
                x = torch.LongTensor(np.array([tokens])).to(cfg.device)

                # Policy
                logits, hidden = policy(x, hidden)
                probs = F.softmax(logits[0, -1], dim=0)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                # Reference policy (for KL penalty)
                with torch.no_grad():
                    ref_logits, ref_hidden = ref_policy(x, ref_hidden)
                    ref_probs = F.softmax(ref_logits[0, -1], dim=0)
                    ref_log_prob = torch.log(ref_probs[action] + 1e-8)

                tokens.append(action.item())
                log_probs.append(log_prob)
                ref_log_probs.append(ref_log_prob)

                if action.item() == 0:  # Stop token
                    break

            # For empty prompts, skip the artificial start token in generated text
            start_idx = 1 if original_prompt_len == 0 else 0
            generated_texts.append(''.join([chr(min(max(t, 0), 127)) for t in tokens[start_idx:]]))
            token_seqs.append(tokens + [0] * (cfg.max_length - len(tokens)))
            log_probs_list.append(torch.stack(log_probs) if log_probs else torch.zeros(1).to(cfg.device))
            ref_log_probs_list.append(torch.stack(ref_log_probs) if ref_log_probs else torch.zeros(1).to(cfg.device))

        # Get rewards from reward model
        x = torch.LongTensor(token_seqs).to(cfg.device)
        with torch.no_grad():
            rewards = reward_model(x)

        # KL penalty (stay close to reference policy)
        kl_penalties = []
        for log_probs, ref_log_probs in zip(log_probs_list, ref_log_probs_list):
            kl = (log_probs - ref_log_probs).sum()
            kl_penalties.append(kl)

        kl_penalties = torch.stack(kl_penalties)
        final_rewards = rewards - cfg.kl_coef * kl_penalties

        # PPO update
        for _ in range(cfg.ppo_epochs):
            # Value targets (Monte Carlo return - no bootstrapping)
            values = value_net(x).mean(dim=1)  # Average over sequence
            advantages = final_rewards - values.detach()

            # Policy loss (PPO-Clip)
            policy_losses = []
            for i, (log_probs, advantage) in enumerate(zip(log_probs_list, advantages)):
                ratio = torch.exp(log_probs - log_probs.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - cfg.ppo_clip_eps, 1 + cfg.ppo_clip_eps) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_losses.append(policy_loss)

            policy_loss = torch.stack(policy_losses).mean()

            # Value loss
            value_loss = F.mse_loss(values, final_rewards.detach())

            # Total loss
            loss = policy_loss + cfg.value_coef * value_loss

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()
            value_optimizer.step()

        if iteration % 10 == 0:
            avg_reward = rewards.mean().item()
            avg_kl = kl_penalties.mean().item()
            console.log(
                f"PPO Iteration {iteration}/{cfg.rlhf_iterations} | "
                f"Reward: {avg_reward:.3f} | KL: {avg_kl:.3f} | "
                f"Sample: '{generated_texts[0][:30]}'"
            )

    console.print("[green]✓ RLHF training complete[/green]\n")


def main():
    parser = argparse.ArgumentParser(description="RLHF for Text Generation")
    parser.add_argument("--task", type=str, default="sentiment",
                       choices=["sentiment", "length", "caps"],
                       help="Task to optimize for")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Iterations per stage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    cfg = Config(
        sft_iterations=args.iterations,
        reward_model_iterations=args.iterations,
        rlhf_iterations=args.iterations,
        seed=args.seed,
        device=args.device,
    )

    # Set seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    console.print(f"[bold green]RLHF Pipeline for Text Generation[/bold green]")
    console.print(f"Task: {args.task}")
    console.print(f"Device: {cfg.device}\n")

    # Initialize models
    policy = PolicyNetwork(cfg.vocab_size, cfg.embed_dim, cfg.hidden_dim).to(cfg.device)
    ref_policy = PolicyNetwork(cfg.vocab_size, cfg.embed_dim, cfg.hidden_dim).to(cfg.device)
    value_net = ValueNetwork(cfg.vocab_size, cfg.embed_dim, cfg.hidden_dim).to(cfg.device)
    reward_model = RewardModel(cfg.vocab_size, cfg.embed_dim, cfg.hidden_dim).to(cfg.device)

    # Stage 1: Supervised Fine-Tuning
    demonstrations = create_synthetic_demonstrations(args.task, num_samples=1000)
    train_sft(policy, demonstrations, cfg)

    # Copy policy for reference (KL penalty)
    ref_policy.load_state_dict(policy.state_dict())

    # Stage 2: Reward Model Training
    preference_pairs = create_preference_pairs(args.task, num_pairs=500)
    train_reward_model(reward_model, preference_pairs, cfg)

    # Stage 3: RLHF with PPO
    train_rlhf_ppo(policy, value_net, reward_model, ref_policy, cfg)

    # Evaluation
    console.print("[bold]Generating samples:[/bold]")
    prompts = ["", "I think", "This is"]

    for prompt in prompts:
        text = policy.generate(prompt, cfg.max_length, temperature=0.8)
        console.print(f"  Prompt: '{prompt}' → '{text}'")

    console.print(f"\n[bold green]✓ RLHF pipeline complete![/bold green]")
    console.print(f"[dim]Note: This is a simplified educational example. Production RLHF uses transformers.[/dim]")


if __name__ == "__main__":
    main()
