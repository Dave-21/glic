const dayContent = {
  T0: {
    image: '../forecasts/forecast_T0.png',
    narrative:
      'Ice remains consolidated across western Lake Superior and northern Lake Huron with thinner new ice pushing south from Georgian Bay.',
    shipping:
      'Key lanes: Duluth to Thunder Bay, St. Marys River downbound tracks remain mostly passable with escort scheduling.',
    fastIce:
      'Fast ice locks to shorelines from Whitefish Bay to Alpena; watch for plate breaks near river mouths during evening warming.'
  },
  T1: {
    image: '../forecasts/forecast_T1.png',
    narrative:
      'Model expands 60–80% concentrations eastward on Superior as northwest winds compact the pack; Erie shows modest opening along the south shore.',
    shipping:
      'Recommend convoy spacing along Superior north shore; Detroit River corridor remains clear for single-hull traffic.',
    fastIce:
      'Fast-ice extent steadies but weakens near Apostle Islands where warm advection erodes the edge.'
  },
  T2: {
    image: '../forecasts/forecast_T2.png',
    narrative:
      'Major operational impact as new pack intrudes into Straits of Mackinac, potentially slowing downbound ore carriers.',
    shipping:
      'Plan early icebreaker staging at St. Ignace and the lower St. Marys locks to handle thicker floes.',
    fastIce:
      'Fast ice fractures along Saginaw Bay with mobile floes drifting toward shipping fairways.'
  },
  T3: {
    image: '../forecasts/forecast_T3.png',
    narrative:
      'Forecast settles into a broad fast-ice fringe with scattered leads on central Superior; Lake Ontario largely clears except near the Bay of Quinte.',
    shipping:
      'Routing priority: maintain corridor between Whitefish Point and the Soo, monitor Niagara peninsula approaches for refreeze.',
    fastIce:
      'Fast ice rebuilds south of Thunder Bay but remains unstable elsewhere—pack intrusions likely wherever winds exceed 20 kt.'
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
