// Cognito's hosted login for the web pages; loaded by index.html and admin.html.
// The hosted login, as the OAuth authorization code flow with PKCE: a page can keep no
// client secret. The access token is held in memory and renewed before it runs low; the
// refresh token sits in this tab's sessionStorage, so a reload keeps the user signed in and
// closing the tab signs them out.
window.login = (() => {
  const domain = window.WEB_LOGIN_DOMAIN, client = window.WEB_CLIENT_ID;
  // Each page that signs in comes back to itself; all are registered as callbacks.
  const page = location.origin + location.pathname;
  let access = '', expiresAt = 0, email = '', groups = [];
  const claimsOf = (jwt) => JSON.parse(atob(jwt.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')));
  const b64url = (bytes) => btoa(String.fromCharCode(...new Uint8Array(bytes)))
    .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

  async function exchange(form) {
    const r = await fetch(domain + '/oauth2/token', {
      method: 'POST', headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: new URLSearchParams({client_id: client, ...form}),
    });
    const got = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error('登录失败：' + (got.error || r.status));
    access = got.access_token;
    expiresAt = Date.now() + (got.expires_in || 3600) * 1000;
    // Which links to show. The router checks the groups on the verified token itself.
    try { groups = claimsOf(access)['cognito:groups'] || []; } catch { groups = []; }
    // Display only: nothing is authorised on the id token, so it is read, not verified.
    try {
      email = claimsOf(got.id_token).email || email;
    } catch { /* no id token on this grant */ }
    if (got.refresh_token) sessionStorage.setItem('refresh', got.refresh_token);
  }

  async function begin() {
    const verifier = b64url(crypto.getRandomValues(new Uint8Array(32)));
    const challenge = b64url(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier)));
    const state = b64url(crypto.getRandomValues(new Uint8Array(16)));
    sessionStorage.setItem('pkce', JSON.stringify({verifier, state}));
    location.assign(domain + '/oauth2/authorize?' + new URLSearchParams({
      response_type: 'code', client_id: client, redirect_uri: page, scope: 'openid email',
      state, code_challenge: challenge, code_challenge_method: 'S256'}));
    return new Promise(() => {});      // the page is leaving
  }

  return {
    async ready() {
      if (!domain || !client) throw new Error('这个部署没有配置网页登录');
      const q = new URLSearchParams(location.search);
      if (q.get('code')) {
        const saved = JSON.parse(sessionStorage.getItem('pkce') || '{}');
        sessionStorage.removeItem('pkce');
        // The code is single-use and must not stay in the address bar or the history.
        history.replaceState(null, '', page + location.hash);
        if (!saved.state || q.get('state') !== saved.state) return begin();
        return exchange({grant_type: 'authorization_code', code: q.get('code'),
                         redirect_uri: page, code_verifier: saved.verifier});
      }
      if (sessionStorage.getItem('refresh')) {
        try { return await this.renew(); } catch { sessionStorage.removeItem('refresh'); }
      }
      return begin();
    },
    renew() {
      return exchange({grant_type: 'refresh_token', refresh_token: sessionStorage.getItem('refresh')});
    },
    // Renewed with half an hour to spare: a consent started with this token has to be
    // completed with this very token, so it must not be about to expire.
    async token() {
      if (Date.now() > expiresAt - 30 * 60 * 1000) await this.renew();
      return access;
    },
    email: () => email,
    groups: () => groups,
    signOut() {
      sessionStorage.clear();
      location.assign(domain + '/logout?' + new URLSearchParams({client_id: client, logout_uri: page}));
    },
  };
})();
