import { render } from '@testing-library/react';
import { it } from 'vitest';
import { MeraPanel } from './MeraPanel';
import { meraFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<MeraPanel frame={meraFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<MeraPanel frame={emptyFrame} />);
});
