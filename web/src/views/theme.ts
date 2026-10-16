// Every colour a canvas draws with, in one place, matching the tokens in
// styles.css. The meaning never changes: blue is a value and the forward
// pass (and "says blue"), orange a gradient and the backward pass (and "says
// orange"), violet a learned parameter.

export type RGB = [number, number, number];

export const BLUE: RGB = [77, 163, 255];
export const ORANGE: RGB = [255, 140, 66];
/** the middle of a blue-orange field: unsure, near zero. Dark, like the page */
export const MID: RGB = [20, 27, 43];
export const VIOLET = '#a78bfa';
export const GREEN = '#d9fdb4';
export const RED = '#ff5d6c';
export const GOLD = '#ffd166';

export const INK = '#e8eef9';
export const INK_2 = '#a3b0c7';
export const MUTED = '#6b7890';
export const GRID = 'rgba(148, 170, 220, 0.09)';
export const AXIS = 'rgba(148, 170, 220, 0.22)';
export const BG = '#070b14';

export const hex = (c: RGB) => `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
export const rgba = (c: RGB, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

/**
 * One colour per optimizer in a comparison: sgd orange, momentum blue,
 * rmsprop pale lime, adam pale pink. Chosen by search so every pair stays far
 * apart for every kind of colour vision on the dark page (colors.test.ts).
 */
export const PALETTE = [hex(ORANGE), hex(BLUE), '#d9fdb4', '#f5bcd9'];

/** layer colours, first to last: violet, lavender, light orange; lighter with depth for every kind of colour vision */
export const LAYER_RAMP: RGB[] = [[100, 80, 200], [180, 130, 240], [255, 210, 135]];

/** a loss landscape: near black at the minimum, dim slate up the walls, so the paths are the brightest thing on it */
export const SURFACE_LOW: RGB = [3, 5, 14];
export const SURFACE_HIGH: RGB = [52, 66, 104];
