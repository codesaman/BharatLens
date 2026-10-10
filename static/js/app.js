/**
 * BharatLens ML - National Portal Web Application
 * Lightweight, modular, accessible client-side logic
 */

let allDistrictsCatalog = [];
let radarChartInstance = null;
let histChartInstance = null;
let shapChartInstance = null;
let stateRankingChartInstance = null;
let stateTrendDonutInstance = null;
let stateTierDonutInstance = null;
let globalShapChartInstance = null;

// Trend & Tier badge helpers with SVG icons (NO EMOJIS)
function getTrendBadgeHtml(trend) {
  trend = trend || 'Stable';
  let badgeClass = 'badge-stable';
  let iconSvg = '';

  if (trend === 'Improving') {
    badgeClass = 'badge-improving';
    iconSvg = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M4 12l1.41 1.41L11 7.83V20h2V7.83l5.58 5.59L20 12l-8-8-8 8z"/></svg>';
  } else if (trend === 'Declining') {
    badgeClass = 'badge-declining';
    iconSvg = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M20 12l-1.41-1.41L13 16.17V4h-2v12.17l-5.58-5.59L4 12l8 8 8-8z"/></svg>';
  } else if (trend === 'Anomalous') {
    badgeClass = 'badge-anomaly';
    iconSvg = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>';
  } else {
    badgeClass = 'badge-stable';
    iconSvg = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M4 9h16v2H4zm0 4h16v2H4z"/></svg>';
  }

  return `<span class="gov-badge ${badgeClass}">${iconSvg} ${trend}</span>`;
}

function getTierBadgeHtml(tier) {
  return `<span class="gov-badge badge-tier">${tier || 'Tier 2'}</span>`;
}

// -----------------------------------------------------------------------------
// 1. Navigation & Tab Routing
// -----------------------------------------------------------------------------
function switchTab(tabId) {
  document.querySelectorAll('.gov-nav-link').forEach(link => {
    link.classList.toggle('active', link.getAttribute('data-tab') === tabId);
  });

  document.querySelectorAll('.portal-tab-content').forEach(section => {
    section.style.display = (section.id === `tab-${tabId}`) ? 'block' : 'none';
  });

  const tabTitles = {
    overview: 'National Overview',
    district: 'District Deep-Dive',
    state: 'State Explorer',
    ml: 'ML Insights & SHAP',
    sources: 'Public Data Sources',
    methodology: 'Methodology & Scoring'
  };
  document.getElementById('breadcrumb-current').textContent = tabTitles[tabId] || 'Overview';
  window.location.hash = tabId;

  // Trigger resize on charts when entering tab
  if (tabId === 'district') {
    if (radarChartInstance) radarChartInstance.resize();
    if (histChartInstance) histChartInstance.resize();
    if (shapChartInstance) shapChartInstance.resize();
  } else if (tabId === 'state') {
    if (stateRankingChartInstance) stateRankingChartInstance.resize();
  } else if (tabId === 'ml') {
    if (globalShapChartInstance) globalShapChartInstance.resize();
  }
}

// -----------------------------------------------------------------------------
// 2. Accessibility Controls
// -----------------------------------------------------------------------------
function setupAccessibility() {
  let currentFontSize = 15;
  document.getElementById('btn-font-inc').addEventListener('click', () => {
    if (currentFontSize < 19) {
      currentFontSize += 1;
      document.documentElement.style.setProperty('--font-base-size', `${currentFontSize}px`);
    }
  });
  document.getElementById('btn-font-dec').addEventListener('click', () => {
    if (currentFontSize > 13) {
      currentFontSize -= 1;
      document.documentElement.style.setProperty('--font-base-size', `${currentFontSize}px`);
    }
  });
  document.getElementById('btn-font-norm').addEventListener('click', () => {
    currentFontSize = 15;
    document.documentElement.style.setProperty('--font-base-size', '15px');
  });

  document.getElementById('btn-contrast-toggle').addEventListener('click', () => {
    document.body.classList.toggle('high-contrast');
  });
}

// -----------------------------------------------------------------------------
// 3. National Overview Initializer
// -----------------------------------------------------------------------------
async function initOverview() {
  try {
    const res = await fetch('/api/summary');
    const data = await res.json();

    document.getElementById('kpi-total-districts').textContent = data.total_districts;
    document.getElementById('kpi-national-score').textContent = data.national_avg_score.toFixed(1);
    document.getElementById('kpi-improving').textContent = data.improving_count;
    document.getElementById('kpi-stable').textContent = data.stable_count;
    document.getElementById('kpi-declining').textContent = data.declining_count;
    document.getElementById('kpi-anomalies').textContent = data.anomaly_count;

    // Populate Top 10 Table
    const topTbody = document.querySelector('#top-districts-table tbody');
    topTbody.innerHTML = '';
    data.top_10_leading.forEach((item, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="fw-bold">${idx + 1}</td>
        <td><strong><a href="#district" class="text-navy text-decoration-none" onclick="inspectDistrictDirect('${item.state}', '${item.district}')">${item.district}</a></strong></td>
        <td>${item.state}</td>
        <td class="fw-bold text-navy">${item.score.toFixed(1)}</td>
        <td class="text-success fw-semibold">+${item.delta.toFixed(1)}</td>
        <td>${getTrendBadgeHtml(item.trend)}</td>
      `;
      topTbody.appendChild(tr);
    });

    // Populate Priority Alert Table
    const alertTbody = document.querySelector('#alert-districts-table tbody');
    alertTbody.innerHTML = '';
    data.top_10_alerts.forEach((item, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="fw-bold">${idx + 1}</td>
        <td><strong><a href="#district" class="text-navy text-decoration-none" onclick="inspectDistrictDirect('${item.state}', '${item.district}')">${item.district}</a></strong></td>
        <td>${item.state}</td>
        <td class="fw-bold">${item.score.toFixed(1)}</td>
        <td class="text-purple fw-bold">${item.anomaly_risk.toFixed(1)}/100</td>
        <td>${getTrendBadgeHtml(item.trend)}</td>
      `;
      alertTbody.appendChild(tr);
    });

    // Render Plotly Pan-India Map
    renderIndiaMap(data.map_points);
  } catch (err) {
    console.error('Error loading summary data:', err);
  }
}

function renderIndiaMap(points) {
  const metricType = document.getElementById('map-metric-select').value;
  const lats = points.map(p => p.lat);
  const lons = points.map(p => p.lon);
  const texts = points.map(p => `<b>${p.district}</b> (${p.state})<br>Score: ${p.score.toFixed(1)}<br>Trend: ${p.trend}<br>Tier: ${p.tier}`);

  let markerColor, colorscale, showscale = true;
  if (metricType === 'delta') {
    markerColor = points.map(p => p.delta);
    colorscale = 'RdYlGn';
  } else if (metricType === 'anomaly') {
    markerColor = points.map(p => p.anomaly_risk);
    colorscale = 'YlOrRd';
  } else if (metricType === 'trend') {
    const trendMap = { 'Improving': 3, 'Stable': 2, 'Declining': 1, 'Anomalous': 0 };
    markerColor = points.map(p => trendMap[p.trend] ?? 2);
    colorscale = [[0, '#6f42c1'], [0.33, '#dc3545'], [0.66, '#0dcaf0'], [1, '#198754']];
    showscale = false;
  } else if (metricType === 'tier') {
    const tierMap = { 0: 0, 1: 1, 2: 2, 3: 3 };
    markerColor = points.map(p => tierMap[p.cluster_id] ?? 1);
    colorscale = 'Viridis';
    showscale = false;
  } else {
    markerColor = points.map(p => p.score);
    colorscale = 'Blues';
  }

  const mapData = [{
    type: 'scattergeo',
    locationmode: 'country names',
    lat: lats,
    lon: lons,
    hoverinfo: 'text',
    text: texts,
    mode: 'markers',
    marker: {
      size: 7,
      color: markerColor,
      colorscale: colorscale,
      showscale: showscale,
      colorbar: { title: 'Score', thickness: 12, len: 0.8 },
      line: { width: 0.5, color: '#ffffff' }
    }
  }];

  const mapLayout = {
    geo: {
      scope: 'asia',
      center: { lat: 22.0, lon: 82.5 },
      projection: { scale: 3.8 },
      showland: true,
      landcolor: '#f8f9fa',
      countrycolor: '#002b49',
      countrywidth: 1.5,
      subunitcolor: '#ced4da',
      showcountries: true,
      showsubunits: true,
      bgcolor: '#ffffff'
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: '#ffffff',
    height: 520
  };

  Plotly.newPlot('india-map-container', mapData, mapLayout, { responsive: true, displayModeBar: false });
}

// -----------------------------------------------------------------------------
// 4. District Deep-Dive Logic
// -----------------------------------------------------------------------------
async function initDistrictSelectors() {
  try {
    const res = await fetch('/api/states');
    const states = await res.json();

    const stateSelect = document.getElementById('select-district-state');
    stateSelect.innerHTML = '';
    states.forEach(st => {
      const opt = document.createElement('option');
      opt.value = st;
      opt.textContent = st;
      stateSelect.appendChild(opt);
    });

    // Default to Maharashtra or first
    if (states.includes('Maharashtra')) {
      stateSelect.value = 'Maharashtra';
    }

    await populateDistrictsForState(stateSelect.value);

    stateSelect.addEventListener('change', async () => {
      await populateDistrictsForState(stateSelect.value);
    });

    document.getElementById('btn-load-district').addEventListener('click', () => {
      const state = stateSelect.value;
      const district = document.getElementById('select-district-name').value;
      loadDistrictDeepDive(state, district);
    });

    // Load initial default district
    loadDistrictDeepDive(stateSelect.value, document.getElementById('select-district-name').value || 'Pune');
  } catch (err) {
    console.error('Error populating district selectors:', err);
  }
}

async function populateDistrictsForState(state) {
  const distSelect = document.getElementById('select-district-name');
  distSelect.innerHTML = '<option value="">Loading...</option>';

  const res = await fetch(`/api/districts?state=${encodeURIComponent(state)}`);
  const districts = await res.json();

  distSelect.innerHTML = '';
  districts.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d.district;
    opt.textContent = d.district;
    distSelect.appendChild(opt);
  });

  // Pick Pune or first
  const match = districts.find(d => d.district.toLowerCase() === 'pune');
  if (match) distSelect.value = match.district;
}

function inspectDistrictDirect(state, district) {
  document.getElementById('select-district-state').value = state;
  populateDistrictsForState(state).then(() => {
    document.getElementById('select-district-name').value = district;
    loadDistrictDeepDive(state, district);
    switchTab('district');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

async function loadDistrictDeepDive(state, district) {
  if (!state || !district) return;
  try {
    const res = await fetch(`/api/district/${encodeURIComponent(state)}/${encodeURIComponent(district)}`);
    const d = await res.json();
    if (d.error) {
      alert(`District data not found: ${district}`);
      return;
    }

    // Update Header Info
    document.getElementById('d-state-label').textContent = d.state;
    document.getElementById('d-title').textContent = d.district;
    document.getElementById('d-pop').textContent = d.population ? Number(d.population).toLocaleString('en-IN') : 'N/A';
    document.getElementById('d-urban').textContent = d.urban_pct ? `${d.urban_pct.toFixed(1)}%` : '0.0%';
    document.getElementById('d-hist-avg').textContent = d.historical_average ? `${d.historical_average.toFixed(1)} pts` : 'N/A';

    // Badges
    document.getElementById('d-badge-tier').textContent = d.cluster_name;
    const trendEl = document.getElementById('d-badge-trend');
    trendEl.outerHTML = getTrendBadgeHtml(d.trend);
    document.getElementById('d-badge-anomaly').style.display = (d.is_anomaly === 1) ? 'inline-flex' : 'none';

    // Summary Score Strips
    document.getElementById('d-score-curr').textContent = d.bharatlens_score.toFixed(1);
    const changeEl = document.getElementById('d-score-change');
    changeEl.textContent = `${d.score_change >= 0 ? '+' : ''}${d.score_change.toFixed(1)} vs NFHS-4`;
    changeEl.className = `small fw-semibold ${d.score_change >= 0 ? 'text-success' : 'text-danger'}`;

    document.getElementById('d-state-avg').textContent = d.state_average.toFixed(1);
    document.getElementById('d-state-diff').textContent = `${d.state_relative_deviation >= 0 ? '+' : ''}${d.state_relative_deviation.toFixed(1)} pts vs State`;

    document.getElementById('d-nat-avg').textContent = d.national_average.toFixed(1);
    document.getElementById('d-nat-diff').textContent = `${d.national_relative_deviation >= 0 ? '+' : ''}${d.national_relative_deviation.toFixed(1)} pts vs India`;
    document.getElementById('d-zscore').textContent = `${d.relative_growth_zscore.toFixed(2)}σ`;

    // Render 5-Dimension Radar Polar Chart
    renderDistrictRadarChart(d.dimensions);

    // Render Historical Paired Bar Chart
    renderHistoricalChart(d.historical_tracking);

    // Update ML Forecast Card
    document.getElementById('ml-pred-score').textContent = `${d.ml_forecast.projected_next_score.toFixed(1)} / 100`;
    const predDeltaEl = document.getElementById('ml-pred-delta');
    predDeltaEl.textContent = `${d.ml_forecast.predicted_delta >= 0 ? '+' : ''}${d.ml_forecast.predicted_delta.toFixed(2)} points predicted momentum`;
    predDeltaEl.className = `small fw-semibold ${d.ml_forecast.predicted_delta >= 0 ? 'text-success' : 'text-danger'}`;

    document.getElementById('ml-pred-class').textContent = d.ml_forecast.predicted_trend;
    document.getElementById('ml-conf-bar').style.width = `${d.ml_forecast.confidence_pct}%`;
    document.getElementById('ml-conf-val').textContent = `${d.ml_forecast.confidence_pct.toFixed(1)}% Confidence`;

    document.getElementById('ml-shap-narrative').innerHTML = d.shap_analysis.narrative.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Render Local SHAP Horizontal Bar Chart
    renderDistrictShapChart(d.shap_analysis.top_positive, d.shap_analysis.top_negative);

    // Populate Peer Districts Table
    const peersTbody = document.querySelector('#district-peers-table tbody');
    peersTbody.innerHTML = '';
    d.peer_districts.forEach(p => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${p.district}</strong></td>
        <td>${p.state}</td>
        <td class="fw-bold">${p.score.toFixed(1)}</td>
        <td class="${p.delta >= 0 ? 'text-success' : 'text-danger'} fw-semibold">${p.delta >= 0 ? '+' : ''}${p.delta.toFixed(1)}</td>
        <td>${getTrendBadgeHtml(p.trend)}</td>
        <td><button type="button" class="btn btn-sm btn-gov-outline py-0" onclick="inspectDistrictDirect('${p.state}', '${p.district}')">View</button></td>
      `;
      peersTbody.appendChild(tr);
    });

  } catch (err) {
    console.error('Error loading district details:', err);
  }
}

function renderDistrictRadarChart(dims) {
  const ctx = document.getElementById('district-radar-chart').getContext('2d');
  if (radarChartInstance) radarChartInstance.destroy();

  radarChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Health & Nutrition', 'Women Empowerment', 'Infrastructure', 'Economic Floor', 'Demographics & Edu'],
      datasets: [{
        label: 'Current Score',
        data: [
          dims.health_nutrition,
          dims.women_empowerment,
          dims.infrastructure,
          dims.economic_employment,
          dims.demographics_education
        ],
        backgroundColor: 'rgba(255, 153, 51, 0.25)',
        borderColor: '#ff9933',
        pointBackgroundColor: '#002b49',
        pointBorderColor: '#ffffff',
        pointHoverBackgroundColor: '#ffffff',
        pointHoverBorderColor: '#002b49',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: { stepSize: 20, font: { size: 9 }, backdropColor: 'transparent' },
          grid: { color: '#e2e8f0' },
          pointLabels: { font: { size: 10, weight: '600' }, color: '#002b49' }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function renderHistoricalChart(tracking) {
  const ctx = document.getElementById('district-hist-chart').getContext('2d');
  if (histChartInstance) histChartInstance.destroy();

  const labels = ['Institutional Births', 'Child Vaccination', 'Female Literacy', 'Electricity Access', 'Improved Sanitation', 'Clean Fuel'];
  const prevData = [
    tracking.institutional_births_prev,
    tracking.children_vaccinated_prev,
    tracking.female_literacy_prev,
    tracking.electricity_access_prev,
    tracking.improved_sanitation_prev,
    tracking.clean_fuel_cooking_prev
  ];
  const currData = [
    tracking.institutional_births_curr,
    tracking.children_vaccinated_curr,
    tracking.female_literacy_curr,
    tracking.electricity_access_curr,
    tracking.improved_sanitation_curr,
    tracking.clean_fuel_cooking_curr
  ];

  histChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'NFHS-4 Baseline (~2015-16)',
          data: prevData,
          backgroundColor: '#94a3b8'
        },
        {
          label: 'NFHS-5 Latest (~2019-21)',
          data: currData,
          backgroundColor: '#006699'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 105,
          title: { display: true, text: 'Observed Rate (%)', font: { size: 11, weight: '600' } },
          grid: { color: '#f1f3f5' }
        },
        x: {
          grid: { display: false },
          ticks: { font: { size: 10 } }
        }
      },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } }
      }
    }
  });
}

function renderDistrictShapChart(positives, negatives) {
  const ctx = document.getElementById('district-shap-chart').getContext('2d');
  if (shapChartInstance) shapChartInstance.destroy();

  const items = [];
  positives.slice().reverse().forEach(p => {
    items.push({ label: p.display_name, val: p.shap_value, color: '#138808' });
  });
  negatives.forEach(n => {
    items.push({ label: n.display_name, val: n.shap_value, color: '#dc3545' });
  });

  shapChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: items.map(i => i.label),
      datasets: [{
        data: items.map(i => i.val),
        backgroundColor: items.map(i => i.color),
        borderRadius: 3
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'SHAP Feature Contribution (Points)', font: { size: 11, weight: '600' } },
          grid: { color: '#f1f3f5' }
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 10 } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(context) {
              const val = context.raw;
              return `Contribution: ${val > 0 ? '+' : ''}${val.toFixed(3)} points`;
            }
          }
        }
      }
    }
  });
}

// -----------------------------------------------------------------------------
// 5. State Explorer Logic
// -----------------------------------------------------------------------------
async function initStateExplorer() {
  try {
    const res = await fetch('/api/states');
    const states = await res.json();

    const stateSelect = document.getElementById('select-state-view');
    stateSelect.innerHTML = '';
    states.forEach(st => {
      const opt = document.createElement('option');
      opt.value = st;
      opt.textContent = st;
      stateSelect.appendChild(opt);
    });

    if (states.includes('Uttar Pradesh')) {
      stateSelect.value = 'Uttar Pradesh';
    }

    document.getElementById('btn-load-state').addEventListener('click', () => {
      loadStateView(stateSelect.value);
    });

    stateSelect.addEventListener('change', () => {
      loadStateView(stateSelect.value);
    });

    loadStateView(stateSelect.value);
  } catch (err) {
    console.error('Error initializing state explorer:', err);
  }
}

async function loadStateView(state) {
  if (!state) return;
  try {
    const res = await fetch(`/api/state/${encodeURIComponent(state)}`);
    const data = await res.json();

    document.getElementById('state-ranking-title-name').textContent = state;
    document.getElementById('st-kpi-count').textContent = data.district_count;
    document.getElementById('st-kpi-mean').textContent = data.state_mean_score.toFixed(1);
    document.getElementById('st-kpi-vs-nat').textContent = `${data.state_vs_national >= 0 ? '+' : ''}${data.state_vs_national.toFixed(1)} vs National Benchmark`;
    document.getElementById('st-kpi-vs-nat').className = `kpi-subtext ${data.state_vs_national >= 0 ? 'text-success' : 'text-danger'}`;

    document.getElementById('st-kpi-lead').textContent = data.leading_district.district;
    document.getElementById('st-kpi-lead-score').textContent = `Score: ${data.leading_district.score.toFixed(1)}`;
    document.getElementById('st-kpi-lag').textContent = data.lagging_district.district;
    document.getElementById('st-kpi-lag-score').textContent = `Score: ${data.lagging_district.score.toFixed(1)}`;

    // Render Trend Donut
    renderStateTrendDonut(data.trend_breakdown);

    // Render Tier Donut
    renderStateTierDonut(data.tier_breakdown);

    // Render State Ranking Bar Chart
    renderStateRankingBarChart(data.districts);

    // Populate State Districts Table
    const tbody = document.querySelector('#state-districts-table tbody');
    tbody.innerHTML = '';
    data.districts.forEach((d, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="fw-bold">${idx + 1}</td>
        <td><strong>${d.district}</strong></td>
        <td class="fw-bold text-navy">${d.score.toFixed(1)}</td>
        <td class="${d.delta >= 0 ? 'text-success' : 'text-danger'} fw-semibold">${d.delta >= 0 ? '+' : ''}${d.delta.toFixed(1)}</td>
        <td>${getTrendBadgeHtml(d.trend)}</td>
        <td>${getTierBadgeHtml(d.tier)}</td>
        <td>${d.is_anomaly === 1 ? '<span class="gov-badge badge-anomaly">Outlier</span>' : '<span class="text-muted small">Normal</span>'}</td>
        <td><button type="button" class="btn btn-sm btn-gov-outline py-0" onclick="inspectDistrictDirect('${state}', '${d.district}')">Inspect</button></td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error('Error loading state view:', err);
  }
}

function renderStateTrendDonut(trendData) {
  const ctx = document.getElementById('state-trend-donut').getContext('2d');
  if (stateTrendDonutInstance) stateTrendDonutInstance.destroy();

  stateTrendDonutInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Improving', 'Stable', 'Declining', 'Anomalous'],
      datasets: [{
        data: [
          trendData['Improving'] || 0,
          trendData['Stable'] || 0,
          trendData['Declining'] || 0,
          trendData['Anomalous'] || 0
        ],
        backgroundColor: ['#138808', '#0dcaf0', '#dc3545', '#6f42c1'],
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } }
      }
    }
  });
}

function renderStateTierDonut(tierData) {
  const ctx = document.getElementById('state-tier-donut').getContext('2d');
  if (stateTierDonutInstance) stateTierDonutInstance.destroy();

  const labels = Object.keys(tierData);
  const vals = Object.values(tierData);

  stateTierDonutInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: vals,
        backgroundColor: ['#ff9933', '#0d6efd', '#0dcaf0', '#138808'],
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } }
      }
    }
  });
}

function renderStateRankingBarChart(districts) {
  const ctx = document.getElementById('state-ranking-chart').getContext('2d');
  if (stateRankingChartInstance) stateRankingChartInstance.destroy();

  // Sort ascending for clean horizontal ranking
  const sorted = districts.slice().sort((a, b) => a.score - b.score);

  stateRankingChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(d => d.district),
      datasets: [{
        label: 'BharatLens Score',
        data: sorted.map(d => d.score),
        backgroundColor: sorted.map(d => {
          if (d.trend === 'Improving') return '#138808';
          if (d.trend === 'Declining') return '#dc3545';
          if (d.trend === 'Anomalous') return '#6f42c1';
          return '#006699';
        }),
        borderRadius: 2
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          min: 0,
          max: 100,
          title: { display: true, text: 'BharatLens Score (0-100)', font: { size: 11, weight: '600' } },
          grid: { color: '#f1f3f5' }
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 10 } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// -----------------------------------------------------------------------------
// 6. ML Insights & Evaluation Initializer
// -----------------------------------------------------------------------------
async function initMlInsights() {
  try {
    const res = await fetch('/api/ml-insights');
    const data = await res.json();

    // Regression Table
    const regTbody = document.querySelector('#ml-reg-table tbody');
    regTbody.innerHTML = '';
    data.regression_benchmarks.forEach(r => {
      const tr = document.createElement('tr');
      const isChamp = r.model_name === data.regression_champion;
      tr.innerHTML = `
        <td><strong>${r.model_name}</strong></td>
        <td>${r.mae.toFixed(4)}</td>
        <td>${r.rmse.toFixed(4)}</td>
        <td class="fw-bold text-navy">${r.r2_score.toFixed(4)}</td>
        <td>${r.cv_r2_mean.toFixed(4)} ± ${r.cv_r2_std.toFixed(4)}</td>
        <td>${isChamp ? '<span class="badge bg-success text-white">CHAMPION (Selected)</span>' : '<span class="text-muted small">Evaluated</span>'}</td>
      `;
      regTbody.appendChild(tr);
    });

    // Classification Table
    const clfTbody = document.querySelector('#ml-clf-table tbody');
    clfTbody.innerHTML = '';
    data.classification_benchmarks.forEach(c => {
      const tr = document.createElement('tr');
      const isChamp = c.model_name === data.classification_champion;
      tr.innerHTML = `
        <td><strong>${c.model_name}</strong></td>
        <td>${(c.accuracy * 100).toFixed(1)}%</td>
        <td>${c.macro_precision.toFixed(4)}</td>
        <td>${c.macro_recall.toFixed(4)}</td>
        <td class="fw-bold text-navy">${c.macro_f1.toFixed(4)}</td>
        <td>${c.cv_f1_macro_mean.toFixed(4)}</td>
        <td>${isChamp ? '<span class="badge bg-success text-white">CHAMPION (Selected)</span>' : '<span class="text-muted small">Evaluated</span>'}</td>
      `;
      clfTbody.appendChild(tr);
    });

    // Confusion Matrix Table
    const cmTbody = document.querySelector('#ml-cm-table tbody');
    cmTbody.innerHTML = '';
    const classNames = ['Improving', 'Stable', 'Declining', 'Anomalous'];
    data.confusion_matrix.forEach((row, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <th class="bg-light text-navy">${classNames[i]}</th>
        <td class="${i === 0 ? 'bg-success-subtle fw-bold' : ''}">${row[0]}</td>
        <td class="${i === 1 ? 'bg-info-subtle fw-bold' : ''}">${row[1]}</td>
        <td class="${i === 2 ? 'bg-danger-subtle fw-bold' : ''}">${row[2]}</td>
        <td class="${i === 3 ? 'bg-warning-subtle fw-bold' : ''}">${row[3]}</td>
      `;
      cmTbody.appendChild(tr);
    });

    // Global SHAP Chart
    renderGlobalShapChart(data.global_shap_importance.slice(0, 12));

    // Cluster Profiles Table
    const cluTbody = document.querySelector('#ml-clusters-table tbody');
    cluTbody.innerHTML = '';
    Object.values(data.cluster_profiles).forEach(p => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="badge bg-secondary">${p.cluster_id}</span></td>
        <td><strong>${p.cluster_name}</strong></td>
        <td>${p.district_count}</td>
        <td>${p.pct_of_all_districts.toFixed(1)}%</td>
        <td class="fw-bold text-navy">${p.mean_bharatlens_score.toFixed(1)}</td>
        <td class="text-success fw-semibold">+${p.mean_score_change.toFixed(1)}</td>
        <td>${p.mean_urban_pct.toFixed(1)}%</td>
      `;
      cluTbody.appendChild(tr);
    });

  } catch (err) {
    console.error('Error loading ML insights:', err);
  }
}

function renderGlobalShapChart(items) {
  const ctx = document.getElementById('ml-global-shap-chart').getContext('2d');
  if (globalShapChartInstance) globalShapChartInstance.destroy();

  const sorted = items.slice().reverse();

  globalShapChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(i => i.display_name),
      datasets: [{
        data: sorted.map(i => i.mean_abs_shap),
        backgroundColor: '#002b49',
        borderRadius: 2
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Mean Absolute SHAP Value (Impact Magnitude)', font: { size: 11, weight: '600' } },
          grid: { color: '#f1f3f5' }
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 10 } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// -----------------------------------------------------------------------------
// 7. Public Data Sources Initializer
// -----------------------------------------------------------------------------
async function initDataSources() {
  try {
    const res = await fetch('/api/data-sources');
    const catalog = await res.json();

    const container = document.getElementById('sources-catalog-container');
    container.innerHTML = '';

    Object.values(catalog).forEach(src => {
      const card = document.createElement('div');
      card.className = 'gov-card mb-3';
      card.innerHTML = `
        <div class="gov-card-header bg-light">
          <h4 class="gov-card-title h6 mb-0">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/></svg>
            ${src.name} (${src.geographic_level})
          </h4>
          <span class="badge bg-secondary small">${src.format}</span>
        </div>
        <div class="gov-card-body">
          <div class="row g-2 small">
            <div class="col-md-4"><strong>Agency / Provider:</strong> ${src.provider}</div>
            <div class="col-md-4"><strong>Time Period:</strong> ${src.time_period}</div>
            <div class="col-md-4"><strong>Update Frequency:</strong> ${src.update_frequency}</div>
            <div class="col-md-12 mt-2">
              <strong>Official Portal:</strong> <a href="${src.portal_url}" target="_blank" rel="noopener" class="text-primary">${src.portal_url}</a>
            </div>
            <div class="col-md-12 mt-1">
              <strong>Important Columns:</strong> ${src.important_columns.map(c => `<code>${c}</code>`).join(' ')}
            </div>
            <div class="col-md-12 mt-1 text-muted">
              <strong>Transformation Pipeline:</strong> ${src.transformation_notes}
            </div>
          </div>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error('Error loading data sources:', err);
  }
}

// -----------------------------------------------------------------------------
// 8. Quick Search Autocomplete Logic
// -----------------------------------------------------------------------------
function setupSearchAutocomplete() {
  const input = document.getElementById('portal-search');
  const dropdown = document.getElementById('search-dropdown');

  input.addEventListener('input', async () => {
    const query = input.value.trim();
    if (query.length < 2) {
      dropdown.style.display = 'none';
      return;
    }

    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
      const results = await res.json();

      if (results.length === 0) {
        dropdown.innerHTML = '<div class="p-2 text-muted small">No matching districts found.</div>';
        dropdown.style.display = 'block';
        return;
      }

      dropdown.innerHTML = '';
      results.slice(0, 10).forEach(r => {
        const item = document.createElement('div');
        item.className = 'search-item d-flex justify-content-between align-items-center';
        item.innerHTML = `
          <div><strong>${r.district}</strong> <span class="text-muted small">(${r.state})</span></div>
          <span class="badge bg-light text-dark border small">Score: ${r.score.toFixed(1)}</span>
        `;
        item.addEventListener('click', () => {
          dropdown.style.display = 'none';
          input.value = `${r.district}, ${r.state}`;
          inspectDistrictDirect(r.state, r.district);
        });
        dropdown.appendChild(item);
      });
      dropdown.style.display = 'block';
    } catch (err) {
      console.error('Error during search:', err);
    }
  });

  document.addEventListener('click', (e) => {
    if (!input.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.style.display = 'none';
    }
  });
}

// -----------------------------------------------------------------------------
// 9. District Comparison Modal Logic
// -----------------------------------------------------------------------------
async function setupComparisonModal() {
  try {
    const res = await fetch('/api/districts');
    const districts = await res.json();

    const d1Select = document.getElementById('compare-d1-select');
    const d2Select = document.getElementById('compare-d2-select');
    d1Select.innerHTML = '';
    d2Select.innerHTML = '';

    districts.forEach(d => {
      const key = `${d.state}___${d.district}`;
      const text = `${d.district} (${d.state})`;
      const opt1 = new Option(text, key);
      const opt2 = new Option(text, key);
      d1Select.add(opt1);
      d2Select.add(opt2);
    });

    // Default select 2 distinct districts (e.g. Pune vs Varanasi)
    const puneOpt = districts.find(d => d.district.toLowerCase() === 'pune');
    const varOpt = districts.find(d => d.district.toLowerCase() === 'varanasi');
    if (puneOpt) d1Select.value = `${puneOpt.state}___${puneOpt.district}`;
    if (varOpt) d2Select.value = `${varOpt.state}___${varOpt.district}`;

    document.getElementById('btn-run-compare').addEventListener('click', async () => {
      const d1 = d1Select.value;
      const d2 = d2Select.value;
      if (!d1 || !d2) return;

      const compareRes = await fetch(`/api/compare?d1=${encodeURIComponent(d1)}&d2=${encodeURIComponent(d2)}`);
      const data = await compareRes.json();
      if (data.error) {
        alert(data.error);
        return;
      }

      // Populate Cards
      document.getElementById('compare-card-1').innerHTML = `
        <div class="gov-card border-top-navy">
          <div class="gov-card-body">
            <h4 class="h5 fw-bold text-navy mb-1">${data.district_1.district}</h4>
            <div class="small text-muted mb-2">${data.district_1.state}</div>
            <div class="display-6 fw-bold text-navy">${data.district_1.score.toFixed(1)}</div>
            <div class="small fw-semibold text-muted">Delta: ${data.district_1.delta >= 0 ? '+' : ''}${data.district_1.delta.toFixed(1)}</div>
            <div class="mt-2">${getTrendBadgeHtml(data.district_1.trend)} ${getTierBadgeHtml(data.district_1.tier)}</div>
          </div>
        </div>
      `;

      document.getElementById('compare-card-2').innerHTML = `
        <div class="gov-card border-top-saffron">
          <div class="gov-card-body">
            <h4 class="h5 fw-bold text-navy mb-1">${data.district_2.district}</h4>
            <div class="small text-muted mb-2">${data.district_2.state}</div>
            <div class="display-6 fw-bold text-navy">${data.district_2.score.toFixed(1)}</div>
            <div class="small fw-semibold text-muted">Delta: ${data.district_2.delta >= 0 ? '+' : ''}${data.district_2.delta.toFixed(1)}</div>
            <div class="mt-2">${getTrendBadgeHtml(data.district_2.trend)} ${getTierBadgeHtml(data.district_2.tier)}</div>
          </div>
        </div>
      `;

      document.getElementById('th-d1-name').textContent = data.district_1.district;
      document.getElementById('th-d2-name').textContent = data.district_2.district;

      const tbody = document.querySelector('#compare-table tbody');
      tbody.innerHTML = '';
      const dims = [
        { label: 'Overall BharatLens Index', k1: data.district_1.score, k2: data.district_2.score },
        { label: 'Health & Nutrition', k1: data.district_1.dimensions.health_nutrition, k2: data.district_2.dimensions.health_nutrition },
        { label: 'Women Empowerment', k1: data.district_1.dimensions.women_empowerment, k2: data.district_2.dimensions.women_empowerment },
        { label: 'Infrastructure & Amenities', k1: data.district_1.dimensions.infrastructure, k2: data.district_2.dimensions.infrastructure },
        { label: 'Economic Floor & Wages', k1: data.district_1.dimensions.economic_employment, k2: data.district_2.dimensions.economic_employment },
        { label: 'Demographics & Education', k1: data.district_1.dimensions.demographics_education, k2: data.district_2.dimensions.demographics_education }
      ];

      dims.forEach(row => {
        const diff = row.k1 - row.k2;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${row.label}</strong></td>
          <td class="fw-bold">${row.k1.toFixed(1)}</td>
          <td class="fw-bold">${row.k2.toFixed(1)}</td>
          <td class="fw-semibold ${diff >= 0 ? 'text-success' : 'text-danger'}">${diff >= 0 ? '+' : ''}${diff.toFixed(1)} pts</td>
        `;
        tbody.appendChild(tr);
      });

      document.getElementById('compare-results').style.display = 'block';
    });
  } catch (err) {
    console.error('Error setting up comparison modal:', err);
  }
}

// -----------------------------------------------------------------------------
// Initial Boot
// -----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  setupAccessibility();
  setupSearchAutocomplete();
  setupComparisonModal();

  // Tab listeners
  document.querySelectorAll('.gov-nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const tabId = link.getAttribute('data-tab');
      switchTab(tabId);
    });
  });

  // Map metric layer changer
  document.getElementById('map-metric-select').addEventListener('change', () => {
    fetch('/api/summary').then(r => r.json()).then(data => renderIndiaMap(data.map_points));
  });

  // Check URL hash
  const initialHash = window.location.hash.replace('#', '') || 'overview';
  switchTab(initialHash);

  // Initialize components
  initOverview();
  initDistrictSelectors();
  initStateExplorer();
  initMlInsights();
  initDataSources();
});
