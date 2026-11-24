"""Analytical probability-space example; no checkpoint or training data."""
from render import plot, np, COLORS, HERE
import json
from pathlib import Path


def render(panel_count):
    py = .6
    relation = np.array([.5, .3, .2])
    qy = np.linspace(.02, .98, 481)
    binary = py*np.log(py/qy)+(1-py)*np.log((1-py)/(1-qy))
    conditional = np.zeros_like(qy)
    teacher = np.r_[py, (1-py)*relation]
    student = np.column_stack([qy, (1-qy[:, None])*relation])
    full = np.sum(teacher*np.log(teacher/student), axis=1)
    assert np.allclose(full, binary+conditional, atol=1e-12)
    # T=1 ordinary softmax: target-logit gradients are q_y-p_y and q_y-1.
    assert np.allclose(np.sum(student, axis=1), 1)
    extra_width = .25 if panel_count == 2 else 0
    fig, axes = plot.subplots(1, panel_count, figsize=(4.3*panel_count + extra_width, 3.65))
    def highlight(ax, x, y, text, color):
        ax.text(x,y,text,transform=ax.transAxes,ha='center',va='center',
                fontsize=11,fontweight='bold',color=color,
                bbox=dict(boxstyle='round,pad=.5',facecolor='white',edgecolor=color,alpha=.95))
    ax = axes[0]
    probs = [teacher, np.r_[.85, .15*relation]]
    x = np.arange(4)
    for offset, probs_i, color, label in zip([-.18, .18], probs, COLORS[:2],
                                           ["Teacher: target mass 0.60", "Student: target mass 0.85"]):
        ax.bar(x+offset, probs_i, width=.34, color=color, label=label)
    ax.set_xticks(x, ["Target y", "Class a", "Class b", "Class c"])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("(a) Distribution example")
    ax.text(.5, .94, "Both conditional distributions: (0.5, 0.3, 0.2)",
            transform=ax.transAxes, ha="center", fontsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1, .87))
    highlight(ax,.65,.43,"Same relations\nDifferent confidence",COLORS[0])
    ax = axes[1]
    ax.plot(qy, full, color=COLORS[0], linewidth=3, label="Full KL")
    ax.plot(qy, binary, color=COLORS[3], linestyle="--", linewidth=1.7, label="Target-mass KL (same curve)")
    ax.plot(qy, conditional, color=COLORS[2], linewidth=2, label="Weighted conditional KL = 0")
    ax.set_title("(b) Supervision decomposition")
    ax.set_ylabel("KL divergence (nats)")
    ax.legend(fontsize=8, frameon=False)
    highlight(ax,.56,.45,"Relations matched\nFull KL remains",COLORS[3])
    if panel_count == 3:
        ax = axes[2]
        ax.axvspan(py, .98, color=COLORS[3], alpha=.09)
        ax.plot(qy, qy-py, color=COLORS[0], linewidth=2, label=r"Full KL: $q_y-p_y$")
        ax.plot(qy, qy-1, color=COLORS[1], linewidth=2, label=r"CE: $q_y-1$")
        ax.plot(qy, conditional, color=COLORS[2], linestyle="--", linewidth=2, label="Conditional KL: 0")
        ax.axhline(0, color="#777777", linewidth=.7)
        ax.set_title("(c) Target-logit gradients")
        ax.set_ylabel(r"Target-logit gradient $\partial L/\partial z_y$")
        highlight(ax,.30,.82,"Opposite gradients\n"+r"$p_y<q_y<1$",COLORS[3])
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    for ax in axes[1:]:
        ax.axvline(py, color="#666666", linestyle=":", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"Student target probability $q_y$ ($p_y=0.60$)")
    fig.tight_layout(pad=.5, w_pad=3.0 if panel_count == 2 else 1.2)
    stem = 'fig_supervision_problem' if panel_count == 3 else 'fig_supervision_problem_ab'
    for ext in ('pdf','png','svg'):
        fig.savefig(HERE/f'{stem}.{ext}',bbox_inches='tight',pad_inches=.025,facecolor='white')
    plot.close(fig)
    Path(__file__).with_name("fig_supervision_problem.json").write_text(json.dumps({
        "kind": "analytical_probability_space_illustration", "temperature": 1,
        "teacher_target_probability": py, "conditional_distribution": relation.tolist(),
        "max_decomposition_error": float(np.max(np.abs(full-binary))),
        "notes": "No all-class standardization; not empirical training evidence."
    }, indent=2)+"\n")


if __name__ == "__main__":
    render(3)
    render(2)
