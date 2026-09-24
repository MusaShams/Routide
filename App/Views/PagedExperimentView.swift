import SwiftUI

struct PagedExperimentView: View {
    @Bindable var llm: LLMEvaluator
    let onRun: () -> Void
    let onCopy: () -> Void
    let onRunInterleavedPolicy: () -> Void
    let onCopyInterleavedPolicy: () -> Void
    let onRunPromptSuite: () -> Void
    let onCopyPromptSuite: () -> Void
    let onNumericalCheck: () -> Void
    let onCopyNumericalCheck: () -> Void
    let onRouteCapture: () -> Void
    let onCopyRouteCapture: () -> Void
    let onHeldOutRoutes: () -> Void
    let onMemoryCampaign: () -> Void
    let onMemoryFollowup: () -> Void
    let onCopyMemoryCampaign: () -> Void

    var body: some View {
        if llm.selectedModel.isPaged {
            VStack(alignment: .leading, spacing: 8) {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Process Memory Campaign", systemImage: "memorychip")
                        .font(.headline)
                    Text("14 fixed requests; 512 vs 576 MiB LRU/None. Includes model loading.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(llm.memoryCampaignStatus)
                        .font(.caption)
                    if llm.runningMemoryCampaign && llm.memoryCampaignRunCount > 0 {
                        ProgressView(
                            value: Double(llm.memoryCampaignProgress),
                            total: Double(llm.memoryCampaignRunCount)
                        )
                    }
                    Button("Run Two-Request Longer-Context Follow-up", action: onMemoryFollowup)
                        .buttonStyle(.borderedProminent)
                        .disabled(llm.running || llm.isLoading || llm.packURL == nil)
                    Text(
                        "Separate memory-only pair; does not retry or replace the stopped campaign."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    HStack {
                        Button("Run Memory Campaign", action: onMemoryCampaign)
                            .buttonStyle(.borderedProminent)
                            .disabled(llm.running || llm.isLoading || llm.packURL == nil)
                        Button("Copy Campaign JSON", action: onCopyMemoryCampaign)
                            .disabled(!llm.canCopyMemoryCampaign)
                    }
                    if let fileURL = llm.memoryCampaignFileURL {
                        ShareLink("Save/Share Campaign JSON", item: fileURL)
                            .disabled(llm.running)
                    }
                }
                Divider()

                HStack {
                    Label("Automated Cold/Warm Experiment", systemImage: "repeat")
                        .font(.headline)
                    Spacer()
                    Stepper(
                        "Repetitions: \(llm.experimentRepetitions)",
                        value: $llm.experimentRepetitions,
                        in: 1 ... 10
                    )
                    .fixedSize()
                    .disabled(llm.running)
                }

                HStack {
                    Text(llm.experimentStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if llm.experimentRunCount > 0 {
                        Text("\(llm.experimentProgress)/\(llm.experimentRunCount)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }

                if llm.running && llm.experimentRunCount > 0 {
                    ProgressView(
                        value: Double(llm.experimentProgress),
                        total: Double(llm.experimentRunCount)
                    )
                }

                HStack {
                    Button(action: onRun) {
                        Label("Run Cold/Warm Experiment", systemImage: "play.circle")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(llm.running || llm.prompt.isEmpty)

                    Button(action: onCopy) {
                        Label("Copy Experiment JSON", systemImage: "doc.on.doc")
                    }
                    .disabled(llm.completedExperiment == nil || llm.running)
                }

                Divider()

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Label(
                                "Balanced Prefetch A/B",
                                systemImage: "arrow.left.arrow.right"
                            )
                            .font(.headline)
                            Text(
                                "None vs "
                                    + llm.interleavedCandidatePrefetchPolicy.title
                            )
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Stepper(
                            "Runs per policy: \(llm.interleavedPolicyRepetitions)",
                            value: $llm.interleavedPolicyRepetitions,
                            in: 2 ... 10,
                            step: 2
                        )
                        .fixedSize()
                        .disabled(llm.running)
                    }

                    HStack {
                        Text(llm.interleavedPolicyStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        if llm.interleavedPolicyRunCount > 0 {
                            Text(
                                "\(llm.interleavedPolicyProgress)/"
                                    + "\(llm.interleavedPolicyRunCount)"
                            )
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                        }
                    }

                    if llm.running && llm.interleavedPolicyRunCount > 0 {
                        ProgressView(
                            value: Double(llm.interleavedPolicyProgress),
                            total: Double(llm.interleavedPolicyRunCount)
                        )
                    }

                    HStack {
                        Button(action: onRunInterleavedPolicy) {
                            Label("Run Balanced A/B", systemImage: "play.circle")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(llm.running || llm.prompt.isEmpty)

                        Button(action: onCopyInterleavedPolicy) {
                            Label("Copy A/B JSON", systemImage: "doc.on.doc")
                        }
                        .disabled(
                            llm.completedInterleavedPolicyExperiment == nil
                                || llm.running
                        )
                    }
                }

                Divider()

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Label(
                                "Five-Category Prefetch Suite",
                                systemImage: "square.grid.2x2"
                            )
                            .font(.headline)
                            Text(
                                "512 MiB LRU • 8 tokens • 2 balanced pairs/category"
                            )
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                        Spacer()
                        if llm.promptSuiteRunCount > 0 {
                            Text(
                                "\(llm.promptSuiteProgress)/"
                                    + "\(llm.promptSuiteRunCount)"
                            )
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                        }
                    }

                    Text(llm.promptSuiteStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    if llm.running && llm.promptSuiteRunCount > 0 {
                        ProgressView(
                            value: Double(llm.promptSuiteProgress),
                            total: Double(llm.promptSuiteRunCount)
                        )
                    }

                    HStack {
                        Button(action: onRunPromptSuite) {
                            Label("Run Prompt Suite", systemImage: "play.circle")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(llm.running)

                        Button(action: onCopyPromptSuite) {
                            Label("Copy Suite JSON", systemImage: "doc.on.doc")
                        }
                        .disabled(llm.completedPromptSuite == nil || llm.running)

                        if let fileURL = llm.promptSuiteFileURL {
                            ShareLink(item: fileURL) {
                                Label(
                                    "Save/Share Suite JSON",
                                    systemImage: "square.and.arrow.up"
                                )
                            }
                            .disabled(llm.running)
                        }
                    }
                }

                Divider()

                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("End-to-End Numerical Check")
                            .font(.headline)
                        Text(llm.numericalCheckStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button(action: onNumericalCheck) {
                        Label("Run Raw Token Check", systemImage: "checkmark.seal")
                    }
                    .disabled(llm.running)

                    Button(action: onCopyNumericalCheck) {
                        Label("Copy Numerical JSON", systemImage: "doc.on.doc")
                    }
                    .disabled(llm.completedNumericalCheck == nil || llm.running)
                }

                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("On-Device Route Capture")
                            .font(.headline)
                        Text(llm.routeCaptureStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button(action: onRouteCapture) {
                        Label(
                            "Capture Routes", systemImage: "point.3.connected.trianglepath.dotted")
                    }
                    .disabled(llm.running || llm.prompt.isEmpty)

                    Button(action: onCopyRouteCapture) {
                        Label("Copy Route JSON", systemImage: "doc.on.doc")
                    }
                    .disabled(llm.completedRouteCapture == nil || llm.running)

                    if let fileURL = llm.routeCaptureFileURL {
                        ShareLink(item: fileURL) {
                            Label("Save/Share Route JSON", systemImage: "square.and.arrow.up")
                        }
                        .disabled(llm.running)
                    }
                }

                Divider()

                VStack(alignment: .leading, spacing: 8) {
                    Text("Held-Out Route Suite")
                        .font(.headline)
                    Text("Five fixed prompts • 128-token cap • 576 MiB LRU • no prefetch")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(llm.heldOutRouteStatus)
                        .font(.caption)
                    if llm.runningHeldOutRouteSuite && llm.heldOutRouteCount > 0 {
                        ProgressView(
                            value: Double(llm.heldOutRouteProgress),
                            total: Double(llm.heldOutRouteCount)
                        )
                    }
                    HStack {
                        Button(action: onHeldOutRoutes) {
                            Label(
                                "Capture Held-Out Routes",
                                systemImage: "point.3.connected.trianglepath.dotted")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(llm.running)
                        if let fileURL = llm.heldOutRouteSuiteFileURL {
                            ShareLink(item: fileURL) {
                                Label(
                                    "Save/Share Held-Out JSON", systemImage: "square.and.arrow.up")
                            }
                            .disabled(llm.running)
                        }
                    }
                }
            }
            .padding(.vertical, 8)
        }
    }
}
