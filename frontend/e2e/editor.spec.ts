import { expect, test, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'

async function openSample(page: Page) {
  await page.goto('/')
  await page.getByTestId('sample').click()
  await expect(page.getByTestId('canvas')).toBeVisible()
  await expect(page.getByTestId('filename')).toHaveText('sample-landscape.png')
  await page.waitForTimeout(800) // let the first tiles arrive
}

async function paint(page: Page, from: [number, number], to: [number, number]) {
  const box = (await page.getByTestId('canvas').boundingBox())!
  await page.mouse.move(box.x + from[0], box.y + from[1])
  await page.mouse.down()
  for (let i = 1; i <= 12; i++) {
    await page.mouse.move(
      box.x + from[0] + ((to[0] - from[0]) * i) / 12,
      box.y + from[1] + ((to[1] - from[1]) * i) / 12,
    )
  }
  await page.mouse.up()
}

test('open sample, brush, remove, compare and export', async ({ page }) => {
  await openSample(page)

  await page.getByTestId('tool-brush').click()
  for (let i = 0; i < 3; i++) await page.keyboard.press(']')
  await paint(page, [330, 470], [850, 470])

  await page.getByTestId('remove').click()
  await expect(page.getByTestId('job-bar')).toBeVisible()
  await expect(page.getByTestId('job-bar')).toBeHidden({ timeout: 60_000 })
  await expect(page.getByTestId('toast')).toHaveCount(0) // shows the error text if the job failed
  await expect(page.getByTestId('history')).toContainText('PatchMatch')
  await expect(page.getByTestId('decision')).toContainText('PatchMatch')

  // undo and redo walk back and forth through the history
  await page.getByTestId('undo').click()
  await expect(page.getByTestId('history').locator('[aria-current="true"]')).toContainText(
    'Original',
  )
  await page.getByTestId('redo').click()
  await expect(page.getByTestId('history').locator('[aria-current="true"]')).toContainText(
    'PatchMatch',
  )

  await page.keyboard.press('c')
  await expect(page.getByTestId('compare')).toHaveAttribute('aria-pressed', 'true')

  await page.getByTestId('export').click()
  await expect(page.getByTestId('export-dialog')).toBeVisible()
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByTestId('save-export').click(),
  ])
  expect(download.suggestedFilename()).toBe('sample-landscape-silgi.png')
  const bytes = readFileSync((await download.path())!)
  expect(bytes.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a') // PNG signature
})

test('reverse alpha removes a known watermark without a selection', async ({ page }) => {
  await openSample(page)
  await page.getByLabel('SAMPLE text').check()
  await page.getByTestId('remove').click()
  await expect(page.getByTestId('job-bar')).toBeHidden({ timeout: 60_000 })
  await expect(page.getByTestId('toast')).toHaveCount(0)
  await expect(page.getByTestId('history')).toContainText('Reverse alpha')
})

test('keyboard shortcuts switch tools and theme toggles', async ({ page }) => {
  await openSample(page)
  await page.keyboard.press('w')
  await expect(page.getByTestId('tool-wand')).toHaveAttribute('aria-pressed', 'true')
  await page.keyboard.press('l')
  await expect(page.getByTestId('tool-lasso')).toHaveAttribute('aria-pressed', 'true')
  await page.keyboard.press('b')
  await expect(page.getByTestId('tool-brush')).toHaveAttribute('aria-pressed', 'true')

  await page.getByTestId('theme').click()
  const theme = await page.evaluate(() => document.documentElement.dataset.theme)
  expect(['light', 'dark']).toContain(theme)
})

test('magic wand selects through the server', async ({ page }) => {
  await openSample(page)
  await page.getByTestId('tool-wand').click()
  const box = (await page.getByTestId('canvas').boundingBox())!
  await page.mouse.click(box.x + 300, box.y + 120) // sky
  await expect(page.getByTestId('undo')).toBeEnabled()
})

test('export dialog explains lossless and lossy formats', async ({ page }) => {
  await openSample(page)
  await page.getByTestId('export').click()
  const dialog = page.getByTestId('export-dialog')
  await expect(dialog).toContainText('PNG and TIFF are lossless')
  await dialog.getByRole('radio', { name: 'JPEG' }).click()
  await expect(dialog).toContainText('JPEG is lossy')
  await expect(page.getByTestId('keep-meta')).not.toBeChecked()
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
})

test('language switches to Turkish', async ({ page }) => {
  await page.goto('/')
  await page.getByLabel('Language').click()
  await expect(page.getByText('Örnek görseli dene')).toBeVisible()
})
