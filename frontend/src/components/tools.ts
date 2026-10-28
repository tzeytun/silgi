import {
  Bandage,
  Brush,
  Circle,
  Eraser,
  Hand,
  Lasso,
  Paintbrush,
  Pentagon,
  Pipette,
  ScanSearch,
  Square,
  Stamp,
  Type,
  WandSparkles,
  type LucideIcon,
} from 'lucide-react'
import type { ToolId } from '../store/useStore'

export const TOOLS: { id: ToolId; icon: LucideIcon; key: string; group: 'selection' | 'manual' }[] =
  [
    { id: 'brush', icon: Brush, key: 'B', group: 'selection' },
    { id: 'eraser', icon: Eraser, key: 'E', group: 'selection' },
    { id: 'rect', icon: Square, key: 'M', group: 'selection' },
    { id: 'ellipse', icon: Circle, key: 'O', group: 'selection' },
    { id: 'lasso', icon: Lasso, key: 'L', group: 'selection' },
    { id: 'polygon', icon: Pentagon, key: 'P', group: 'selection' },
    { id: 'wand', icon: WandSparkles, key: 'W', group: 'selection' },
    { id: 'color', icon: Pipette, key: 'K', group: 'selection' },
    { id: 'text', icon: Type, key: 'T', group: 'selection' },
    { id: 'similar', icon: ScanSearch, key: 'F', group: 'selection' },
    { id: 'source', icon: Paintbrush, key: 'G', group: 'manual' },
    { id: 'clone', icon: Stamp, key: 'S', group: 'manual' },
    { id: 'heal', icon: Bandage, key: 'H', group: 'manual' },
    { id: 'hand', icon: Hand, key: 'Y', group: 'manual' },
  ]
