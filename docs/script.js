const dayContent = {
  T0: {
    image: '../forecasts/forecast_T0.png',
    narrative:
      'Baseline GLSEA analysis plus the latest NIC chart define the initial concentration field. Update this text with the daily briefing.',
    shipping:
      'Verify Duluth–Thunder Bay and St. Marys traffic corridors against the buffered route mask before publishing.',
    fastIce:
      'Fast-ice annotations rely on shoreline buffers and the land mask. Confirm any edits prior to release.'
  },
  T1: {
    image: '../forecasts/forecast_T1.png',
    narrative:
      'Review HRRR-driven guidance for the first forecast day and adjust language if external intel contradicts the model.',
    shipping:
      'Note any shipping restrictions triggered by pack motion or rapid concentration changes.',
    fastIce:
      'Call out shore-fast segments that may detach within 24 hours.'
  },
  T2: {
    image: '../forecasts/forecast_T2.png',
    narrative:
      'Summarize anticipated consolidation or breakup events two days out based on the model state and regional climatology.',
    shipping:
      'Document any escort requirements or waypoints that should be monitored for shear ridging.',
    fastIce:
      'Highlight bays or river mouths where the fast-ice mask indicates likely fracture lines.'
  },
  T3: {
    image: '../forecasts/forecast_T3.png',
    narrative:
      'Provide a conservative three-day outlook noting confidence limits and any areas requiring SAR confirmation.',
    shipping:
      'State the highest-risk corridors for day three along with any recommended staging locations.',
    fastIce:
      'Clarify which fast-ice sections are expected to persist through day three.'
  }
};

function buildPlaceholder(label, accent) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500">
      <defs>
        <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="${accent}" stop-opacity="0.85" />
          <stop offset="100%" stop-color="#050c16" stop-opacity="0.95" />
        </linearGradient>
      </defs>
      <rect width="800" height="500" fill="url(#grad)" rx="24" />
      <text x="50%" y="45%" text-anchor="middle" font-size="64" fill="#ffffff" font-family="Inter, sans-serif">
        ${label}
      </text>
      <text x="50%" y="62%" text-anchor="middle" font-size="28" fill="#ffffff" opacity="0.8" font-family="Inter, sans-serif">
        Drop rendered PNGs into /forecasts to override this preview
      </text>
    </svg>
  `;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

const fallbackImages = {
  T0: buildPlaceholder('T0 • Initial', '#4ad6ff'),
  T1: buildPlaceholder('T+1 Outlook', '#86f0b6'),
  T2: buildPlaceholder('T+2 Impacts', '#f5a623'),
  T3: buildPlaceholder('T+3 Pattern', '#ff7ab5')
};

const dayButtons = document.querySelectorAll('.day-buttons button');
const narrativeText = document.getElementById('narrativeText');
const shippingInsight = document.getElementById('shippingInsight');
const fastIceInsight = document.getElementById('fastIceInsight');
const forecastImage = document.getElementById('forecastImage');
const shippingOverlay = document.getElementById('shippingOverlay');
const fastIceOverlay = document.getElementById('fastIceOverlay');
const shippingToggle = document.getElementById('shippingToggle');
const fastIceToggle = document.getElementById('fastIceToggle');

function updateDay(dayKey) {
  dayButtons.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.day === dayKey);
  });

  const content = dayContent[dayKey];
  forecastImage.onerror = null;
  forecastImage.src = content.image;
  forecastImage.alt = `Forecast for ${dayKey}`;
  forecastImage.onerror = () => {
    forecastImage.onerror = null;
    forecastImage.src = fallbackImages[dayKey];
  };
  narrativeText.textContent = content.narrative;
  shippingInsight.textContent = content.shipping;
  fastIceInsight.textContent = content.fastIce;
}

shippingToggle.addEventListener('change', (event) => {
  shippingOverlay.classList.toggle('hidden', !event.target.checked);
});

fastIceToggle.addEventListener('change', (event) => {
  fastIceOverlay.classList.toggle('hidden', !event.target.checked);
});

dayButtons.forEach((button) => {
  button.addEventListener('click', () => updateDay(button.dataset.day));
});

updateDay('T0');
