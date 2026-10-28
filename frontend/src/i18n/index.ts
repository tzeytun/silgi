import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import en from './en'
import tr from './tr'

export type Lang = 'en' | 'tr'

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem('silgi.lang')
    if (saved === 'en' || saved === 'tr') return saved
  } catch {
    /* storage unavailable */
  }
  return navigator.language?.toLowerCase().startsWith('tr') ? 'tr' : 'en'
}

void i18n.use(initReactI18next).init({
  resources: { en: { translation: en }, tr: { translation: tr } },
  lng: initialLang(),
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
})

export function setLang(lang: Lang): void {
  void i18n.changeLanguage(lang)
  document.documentElement.lang = lang
  try {
    localStorage.setItem('silgi.lang', lang)
  } catch {
    /* ignore */
  }
}

document.documentElement.lang = i18n.language

export default i18n
