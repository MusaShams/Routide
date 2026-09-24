// Copyright © 2025 Apple Inc.

import Foundation

struct PresetPrompt: Identifiable {
    let id = UUID()
    let prompt: String
    let enableTools: Bool
    let enableThinking: Bool
    let isLongPrompt: Bool

    init(
        _ prompt: String, enableTools: Bool = false, enableThinking: Bool = false,
        isLongPrompt: Bool = false
    ) {
        self.prompt = prompt
        self.enableTools = enableTools
        self.enableThinking = enableThinking
        self.isLongPrompt = isLongPrompt
    }
}

struct PresetPrompts {
    static let all: [PresetPrompt] = [
        PresetPrompt("Why is the sky blue?"),
        PresetPrompt("What would a medieval knight's Yelp review of a dragon's lair look like?"),
        PresetPrompt("Explain why socks disappear in the dryer from the dryer's perspective."),

        PresetPrompt(
            "Write a breaking news report about cats discovering they can vote.",
            enableThinking: true),
        PresetPrompt(
            "Write a performance review for the person whose job is to make sure Mondays feel terrible.",
            enableThinking: true),

        PresetPrompt("What's the weather in Paris?", enableTools: true),
        PresetPrompt("What is the current time?", enableTools: true),

        PresetPrompt(loadPrompt(named: "LongPrompt"), enableThinking: true, isLongPrompt: true),
        PresetPrompt(loadPrompt(named: "CarKeysStory"), isLongPrompt: true),
    ]
}

struct PrefetchGeneralizationPrompt: Encodable, Sendable {
    let id: String
    let category: String
    let text: String
}

enum PrefetchGeneralizationPrompts {
    static let corpusID = "routide-routing-v1"

    static let all = [
        PrefetchGeneralizationPrompt(
            id: "conversation-002",
            category: "conversation",
            text:
                "Help me plan a quiet weekend at home that balances rest, household chores, "
                + "and one creative activity. Explain why the schedule is realistic."
        ),
        PrefetchGeneralizationPrompt(
            id: "code-002",
            category: "code",
            text:
                "Review this Python function and provide a corrected implementation that "
                + "preserves order while removing duplicates: def unique(values): return "
                + "list(set(values)). Explain the behavioral bug and test edge cases."
        ),
        PrefetchGeneralizationPrompt(
            id: "mathematics-002",
            category: "mathematics",
            text:
                "Find all real solutions of x^4 - 5x^2 + 4 = 0. Explain the substitution "
                + "you use and verify the solutions in the original equation."
        ),
        PrefetchGeneralizationPrompt(
            id: "reasoning-002",
            category: "reasoning",
            text:
                "A device becomes slower after ten minutes, but memory use remains flat. "
                + "Give three competing hypotheses, the measurement that would distinguish "
                + "each one, and the order in which you would test them."
        ),
        PrefetchGeneralizationPrompt(
            id: "expository-002",
            category: "expository",
            text:
                "Explain why many small random reads from flash storage can be slower than "
                + "fewer large sequential reads even when the total byte count is identical."
        ),
    ]
}
