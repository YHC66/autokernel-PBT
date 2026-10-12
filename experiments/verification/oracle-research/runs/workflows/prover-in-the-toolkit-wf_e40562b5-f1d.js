export const meta = {
  name: 'prover-in-the-toolkit',
  description: 'Run 16 oracle-research agents (tests-only vs tests+Lean) and 8 adherence-proof agents in sealed sandboxes',
  phases: [
    { title: 'Oracle research', detail: '2 tasks x 2 arms x 4 replicates' },
    { title: 'Adherence proofs', detail: '4 kernels x 2 replicates, core Lean 4' },
  ],
}

const ROOT = '<scratchpad>/runs'

const ORACLE_SCHEMA = {
  type: 'object',
  properties: {
    sandbox: { type: 'string' },
    task: { type: 'string' },
    arm: { type: 'string' },
    claims_complete: { type: 'boolean', description: 'whether RESULT.md claims the completed bundle characterizes the reference' },
    claim_grounds: { type: 'string' },
    admitted_families: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, description: { type: 'string' }, file: { type: 'string' } }, required: ['name', 'description'] } },
    added_laws: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, statement: { type: 'string' }, tolerance: { type: 'string' } }, required: ['name', 'statement'] } },
    correct_variants_checked: { type: 'array', items: { type: 'string' } },
    lean_file: { type: 'string', description: 'path of the Lean file, or empty if none' },
    lean_main_theorem: { type: 'string' },
    lean_compiled: { type: 'boolean' },
    primitive_assumptions: { type: 'array', items: { type: 'string' }, description: 'every assumption made about exp/sqrt/log etc., one statement each' },
    statement_vs_laws: { type: 'string', description: 'how the Lean statement differs from what laws.py checks' },
    remarks_on_formal_ambiguity: { type: 'string' },
    elapsed_seconds: { type: 'number' },
    outside_paths_read: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['sandbox', 'task', 'arm', 'claims_complete', 'claim_grounds', 'admitted_families', 'added_laws', 'elapsed_seconds', 'outside_paths_read'],
}

const ADHERENCE_SCHEMA = {
  type: 'object',
  properties: {
    sandbox: { type: 'string' },
    kernel: { type: 'string' },
    verdict: { type: 'string', enum: ['adheres', 'refuted', 'undecided'] },
    refutation_kind: { type: 'string', description: 'lean-negation | concrete-input | none' },
    files: { type: 'array', items: { type: 'string' } },
    theorem_statement: { type: 'string' },
    compiled: { type: 'boolean' },
    print_axioms_line: { type: 'string' },
    primitive_assumptions: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, statement: { type: 'string' }, justification: { type: 'string' } }, required: ['name', 'statement'] } },
    model_description: { type: 'string', description: 'what M computes, in words, and how it mirrors kernel.py' },
    not_established: { type: 'string' },
    elapsed_seconds: { type: 'number' },
    outside_paths_read: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['sandbox', 'kernel', 'verdict', 'files', 'compiled', 'primitive_assumptions', 'elapsed_seconds', 'outside_paths_read'],
}

const oraclePrompt = (dir, task, arm) => `You are a research agent. Your working directory, and the ONLY directory you may read, write or search in, is:

${dir}

Start by running \`date +%s\` and then reading ${dir}/README.md; it is the complete task statement, and it is authoritative. Follow it to the letter. Use the interpreter ${dir}/venv/bin/python for all Python. Do not read, search or open anything outside that directory; do not use the network. Work until you have a result you can defend, with every claim demonstrated by a file in the directory. Write RESULT.md as the README asks, run \`date +%s\` again, and then return the structured report (task "${task}", arm "${arm}", sandbox "${dir}"). The structured report is the return value; RESULT.md is the human record. Both must agree.`

const adherencePrompt = (dir, kernel) => `You are a verification agent. Your working directory, and the ONLY directory you may read, write or search in, is:

${dir}

Start by running \`date +%s\` and then reading ${dir}/README.md; it is the complete task statement, and it is authoritative. Follow it to the letter. Use the interpreter ${dir}/venv/bin/python for all Python and the pinned Lean binary named in ${dir}/proof/README.md for all Lean. Do not read, search or open anything outside that directory; do not use the network. Work until your verdict is certified as the README requires. Write RESULT.md, run \`date +%s\` again, and then return the structured report (kernel "${kernel}", sandbox "${dir}"). Both must agree.`

const oracleCells = []
for (const task of ['softmax', 'layernorm'])
  for (const arm of ['T', 'F'])
    for (const r of [1, 2, 3, 4])
      oracleCells.push({ task, arm, r, dir: `${ROOT}/oracle/${task}-${arm}-r${r}` })

const adherenceCells = []
for (const k of ['k1', 'k2', 'k3', 'k4'])
  for (const r of [1, 2])
    adherenceCells.push({ k, r, dir: `${ROOT}/adherence/${k}-r${r}` })

log(`launching ${oracleCells.length} oracle-research agents and ${adherenceCells.length} adherence agents`)

const results = await parallel([
  ...oracleCells.map(c => () =>
    agent(oraclePrompt(c.dir, c.task, c.arm), { label: `oracle:${c.task}-${c.arm}-r${c.r}`, phase: 'Oracle research', schema: ORACLE_SCHEMA })
      .then(x => ({ kind: 'oracle', cell: c, report: x }))),
  ...adherenceCells.map(c => () =>
    agent(adherencePrompt(c.dir, c.k), { label: `adherence:${c.k}-r${c.r}`, phase: 'Adherence proofs', schema: ADHERENCE_SCHEMA })
      .then(x => ({ kind: 'adherence', cell: c, report: x }))),
])

const done = results.filter(Boolean)
log(`${done.length}/${results.length} agents returned a report`)
return { oracle: done.filter(x => x.kind === 'oracle'), adherence: done.filter(x => x.kind === 'adherence'), missing: results.length - done.length }