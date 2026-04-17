/** @typedef {import('@lib/links').Link} Link */

/**
 * @typedef {Object} SiteConfig
 * @property {string} SiteConfig.title
 * @property {string} SiteConfig.description
 * @property {URL} SiteConfig.domain
 * @property {Date} SiteConfig.copyright
 * @property {string} [SiteConfig.ogImage]
 * @property {string} [SiteConfig.favicon]
 */

/** @type {SiteConfig} */
export const SITE = {
  title: 'Conceptual Misalignment',
  description: 'A network graph representation of conceptual misalignment.',
  domain: new URL('https://conceptual-misalignment.netlify.app/'),
  copyright: new Date('2026-04-17'),
  ogImage: undefined,
  // favicon: '/favicon.svg',
}

/** @type {(string | URL)[]} */
export const SOCIALS = []
