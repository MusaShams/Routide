// Copyright © 2025 Apple Inc.

import RoutideMLXRuntime
import RoutideRuntime
import SwiftUI

struct HeaderView: View {
    @Bindable var llm: LLMEvaluator
    @Binding var selectedDisplayStyle: ContentView.DisplayStyle

    @Environment(\.horizontalSizeClass) var horizontalSizeClass

    var status: some View {
        // Model info with status
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Model")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Text(llm.modelInfo)
                    .font(.headline)
                    .lineLimit(1)
            }

            Spacer()

            if llm.running {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Generating...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    var options: some View {
        HStack(spacing: 24) {
            Picker(
                "Model",
                selection: Binding(
                    get: { llm.selectedModel },
                    set: { llm.selectModel($0) }
                )
            ) {
                ForEach(BenchmarkModel.allCases) { model in
                    Text(model.title).tag(model)
                }
            }
            .disabled(!llm.canSelectModel)
            .frame(maxWidth: 260)

            Toggle("Tools", isOn: $llm.includeWeatherTool)
                .toggleStyle(.switch)
                .fixedSize()
                .help("Enable function calling with weather, math, and time tools")
                .disabled(llm.selectedModel.isPaged)

            Toggle("Thinking", isOn: $llm.enableThinking)
                .toggleStyle(.switch)
                .fixedSize()
                .help("Enable thinking mode (supported by Qwen3)")
                .disabled(llm.selectedModel.isPaged)
        }
    }

    @ViewBuilder
    var pagedOptions: some View {
        if llm.selectedModel.isPaged {
            HStack(spacing: 12) {
                Button("Select Expert Pack...") {
                    llm.showingPackImporter = true
                }
                .disabled(llm.running)

                Picker(
                    "Cache",
                    selection: Binding(
                        get: { llm.pagedCacheBudgetBytes },
                        set: { llm.updatePagedCacheBudget($0) }
                    )
                ) {
                    Text("64 MiB").tag(64 * 1024 * 1024)
                    Text("128 MiB").tag(128 * 1024 * 1024)
                    Text("256 MiB").tag(256 * 1024 * 1024)
                    Text("512 MiB").tag(512 * 1024 * 1024)
                    Text("576 MiB").tag(576 * 1024 * 1024)
                    Text("640 MiB").tag(640 * 1024 * 1024)
                    Text("768 MiB").tag(768 * 1024 * 1024)
                    Text("1 GiB").tag(1024 * 1024 * 1024)
                }
                .frame(maxWidth: 150)
                .disabled(llm.running)

                Picker(
                    "Policy",
                    selection: Binding(
                        get: { llm.pagedCachePolicy },
                        set: { llm.updatePagedCachePolicy($0) }
                    )
                ) {
                    Text("LRU").tag(ExpertCachePolicy.lru)
                    Text("Recency-Frequency").tag(ExpertCachePolicy.hybrid)
                }
                .frame(maxWidth: 210)
                .disabled(llm.running)

                Picker(
                    "Prefetch",
                    selection: Binding(
                        get: { llm.pagedPrefetchPolicy },
                        set: { llm.updatePagedPrefetchPolicy($0) }
                    )
                ) {
                    ForEach(ExpertPrefetchPolicy.allCases, id: \.self) { policy in
                        Text(policy.title).tag(policy)
                    }
                }
                .frame(maxWidth: 190)
                .disabled(llm.running)

                Toggle("Cold cache", isOn: $llm.pagedColdCacheBeforeRun)
                    .toggleStyle(.switch)
                    .fixedSize()

                Text(llm.packStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
    }

    var tokens: some View {
        // Max tokens slider
        VStack(alignment: .leading, spacing: 4) {
            Text("Max Tokens: \(llm.maxTokens)")
                .font(.caption)
                .foregroundStyle(.secondary)

            Slider(
                value: Binding(
                    get: { log2(Double(llm.maxTokens)) },
                    set: { llm.maxTokens = Int(pow(2, $0)) }
                ),
                in: llm.selectedModel.isPaged ? 0 ... 9 : 10 ... 15,
                step: 1
            )
            .frame(width: 120)
            .help(
                llm.selectedModel.isPaged
                    ? "Maximum paged tokens to generate (1-512)"
                    : "Maximum number of tokens to generate (1024-32768)"
            )
        }
    }

    var display: some View {
        Picker("Display", selection: $selectedDisplayStyle) {
            ForEach(ContentView.DisplayStyle.allCases, id: \.self) { option in
                Text(option.rawValue.capitalized)
                    .tag(option)
            }

        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(maxWidth: 180)
    }

    var compactControls: some View {
        VStack(alignment: .leading, spacing: 14) {
            Picker(
                "Model",
                selection: Binding(
                    get: { llm.selectedModel },
                    set: { llm.selectModel($0) }
                )
            ) {
                ForEach(BenchmarkModel.allCases) { model in
                    Text(model.title).tag(model)
                }
            }
            .pickerStyle(.menu)
            .disabled(!llm.canSelectModel)
            .frame(maxWidth: .infinity, alignment: .leading)

            if llm.selectedModel.isPaged {
                Button("Select Expert Pack...") {
                    llm.showingPackImporter = true
                }
                .disabled(llm.running)

                Picker(
                    "Expert Cache",
                    selection: Binding(
                        get: { llm.pagedCacheBudgetBytes },
                        set: { llm.updatePagedCacheBudget($0) }
                    )
                ) {
                    Text("64 MiB").tag(64 * 1024 * 1024)
                    Text("128 MiB").tag(128 * 1024 * 1024)
                    Text("256 MiB").tag(256 * 1024 * 1024)
                    Text("512 MiB").tag(512 * 1024 * 1024)
                    Text("576 MiB").tag(576 * 1024 * 1024)
                    Text("640 MiB").tag(640 * 1024 * 1024)
                    Text("768 MiB").tag(768 * 1024 * 1024)
                    Text("1 GiB").tag(1024 * 1024 * 1024)
                }
                .pickerStyle(.menu)
                .disabled(llm.running)

                Picker(
                    "Cache Policy",
                    selection: Binding(
                        get: { llm.pagedCachePolicy },
                        set: { llm.updatePagedCachePolicy($0) }
                    )
                ) {
                    Text("LRU").tag(ExpertCachePolicy.lru)
                    Text("Recency-Frequency").tag(ExpertCachePolicy.hybrid)
                }
                .pickerStyle(.menu)
                .disabled(llm.running)

                Picker(
                    "Prefetch",
                    selection: Binding(
                        get: { llm.pagedPrefetchPolicy },
                        set: { llm.updatePagedPrefetchPolicy($0) }
                    )
                ) {
                    ForEach(ExpertPrefetchPolicy.allCases, id: \.self) { policy in
                        Text(policy.title).tag(policy)
                    }
                }
                .pickerStyle(.menu)
                .disabled(llm.running)

                Toggle("Cold cache", isOn: $llm.pagedColdCacheBeforeRun)
                    .toggleStyle(.switch)

                Text(llm.packStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                HStack(spacing: 20) {
                    Toggle("Tools", isOn: $llm.includeWeatherTool)
                    Toggle("Thinking", isOn: $llm.enableThinking)
                }
                .toggleStyle(.switch)
            }

            tokens
                .frame(maxWidth: .infinity, alignment: .leading)
            display
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.top, 10)
    }

    var body: some View {
        if horizontalSizeClass == .compact {
            VStack {
                status
                DisclosureGroup("Controls") {
                    compactControls
                }
            }
        } else {
            VStack(alignment: .leading, spacing: 12) {
                status

                // Controls row
                HStack(spacing: 16) {
                    HStack(spacing: 24) {
                        options
                        tokens
                    }
                    pagedOptions

                    Spacer()

                    display
                }
            }
            .padding(.bottom, 12)
        }
    }
}
