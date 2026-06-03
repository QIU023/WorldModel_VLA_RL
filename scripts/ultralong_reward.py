"""Ultra-Long dense sub-goal reward —— ultra_long 无 demo,用 BDDL goal 的 And-predicate 子目标计数做 dense progress。
接地真任务结构(kitchen 5 子目标 → 0/.2/.../1.0),给想象 RL / reward-model 训练当 grounded 进度信号。
"""

def goal_predicates(env):
    """返回 ultra_long 任务的 goal predicate 列表(每个 = [pred_name, obj1, (obj2)])。"""
    return env.env.parsed_problem["goal_state"]


def subgoal_progress(env):
    """返回 (满足的子目标数, 总子目标数)。dense progress = sat/total ∈ [0,1]。"""
    gs = env.env.parsed_problem["goal_state"]
    sat = 0
    for s in gs:
        try:
            if env.env._eval_predicate(s):
                sat += 1
        except Exception:
            pass
    return sat, len(gs)


def progress_fraction(env):
    sat, tot = subgoal_progress(env)
    return sat / max(1, tot)
