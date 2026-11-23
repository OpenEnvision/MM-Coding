// Model-call budget barrier for the pinned DSH runtime; no model-facing tool/context.
// Loaded as a local Cordis plugin by worker_rv.py. agent/request is awaited BEFORE the provider
// stream, so the request after the last budgeted call is never sent.
import fs from 'node:fs/promises';
import path from 'node:path';

export const name = 'rv-call-budget';

async function writeAtomic(file, value) {
  const temp = `${file}.${process.pid}.tmp`;
  await fs.writeFile(temp, JSON.stringify(value), { mode: 0o600 });
  await fs.rename(temp, file);
}

export async function apply(ctx) {
  const root = process.env.OSWORLD_DSH_ATTEMPT;
  if (!root) throw new Error('RV_CONTROL_CONFIGURATION: missing attempt');
  const config = JSON.parse(await fs.readFile(path.join(root, 'call_budget.json'), 'utf8'));
  const offset = config.call_offset || 0;
  const started = Date.now();
  const completed = agent => {
    const events = agent.session.snapshotEvents?.() ?? agent.session.events;
    if (!Array.isArray(events)) throw new Error('RV_CONTROL_CONFIGURATION: unsupported session event API');
    return events.filter(e => e.type === 'step/end').length;
  };

  ctx.on('agent/request', async ({ agent }, next) => {
    const done = completed(agent);
    if (offset + done >= config.call_budget) {
      await writeAtomic(path.join(root, 'budget.json'), { budget: config.call_budget,
        model_calls: offset + done, local_model_calls: done, exhausted: true,
        elapsed_s: (Date.now() - started) / 1000, enforcement: 'synchronous-before-provider-request' });
      throw new Error(`RV_BUDGET_EXHAUSTED: Model call budget exhausted (${config.call_budget} calls); TIMEOUT`);
    }
    const local = done + 1;
    await writeAtomic(path.join(root, 'budget.json'), { budget: config.call_budget,
      model_calls: offset + local, local_model_calls: local, exhausted: false,
      elapsed_s: (Date.now() - started) / 1000,
      enforcement: 'synchronous-before-provider-request' });
    return next();
  });
}
