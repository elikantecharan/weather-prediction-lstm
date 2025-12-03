// scripts.js - Frontend Logic & Chart Renderer
const state = {
    forecastDays: 1,
    isLoading: false
};

const $ = id => document.getElementById(id);

let tempChartInstance = null;
let humChartInstance = null;
let windChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    // Set today's date as default
    const today = new Date().toISOString().split('T')[0];
    const dateInput = $('forecastDate');
    if (dateInput) dateInput.value = today;

    // Range buttons toggle
    document.querySelectorAll('.range-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.forecastDays = parseInt(btn.dataset.days);
        });
    });

    // Predict button listener
    const predictBtn = $('predictBtn');
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePredict);
    }

    // Render initial charts
    initCharts(1, [28.9], [60], [18.5]);
});

function collectFormData() {
    return {
        location: $('location').value.trim() || 'Hyderabad, India',
        date: $('forecastDate').value || new Date().toISOString().split('T')[0],
        temperature: parseFloat($('temperature').value) || 28,
        humidity: parseFloat($('humidity').value) || 60,
        wind_speed: parseFloat($('windSpeed').value) || 18,
        pressure: parseFloat($('pressure').value) || 1013,
        precipitation: parseFloat($('precipitation').value) || 0,
        condition: $('weatherCondition').value || 'sunny',
        forecast_days: state.forecastDays
    };
}

async function handlePredict() {
    if (state.isLoading) return;
    const data = collectFormData();

    setLoading(true);

    try {
        const response = await fetchPrediction(data);
        displayResults(response, data);
        updateCharts(response, data);
    } catch (e) {
        console.warn('Backend offline — using analytical simulation data.', e);
        const simulated = simulatePrediction(data);
        displayResults(simulated, data);
        updateCharts(simulated, data);
    } finally {
        setLoading(false);
    }
}

async function fetchPrediction(data) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);

    const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
        signal: controller.signal
    });

    clearTimeout(timeout);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
}

function simulatePrediction(data) {
    const days = data.forecast_days;
    const baseTemp = data.temperature;
    const baseHum = data.humidity;
    const baseWind = data.wind_speed;

    const tempSeries = [];
    const humSeries = [];
    const windSeries = [];

    for (let i = 0; i < days; i++) {
        const deltaT = (Math.random() - 0.45) * 1.8;
        const deltaH = (Math.random() - 0.5) * 4;
        const deltaW = (Math.random() - 0.5) * 3;

        tempSeries.push(Math.round((baseTemp + deltaT + i * 0.2) * 10) / 10);
        humSeries.push(Math.round(Math.max(30, Math.min(95, baseHum + deltaH))));
        windSeries.push(Math.round(Math.max(5, Math.min(45, baseWind + deltaW)) * 10) / 10);
    }

    const rainProb = data.condition === 'rainy' || data.condition === 'storm' 
        ? Math.round(65 + Math.random() * 30) 
        : Math.round(10 + Math.random() * 25);

    const condMap = {
        sunny: { label: 'Sunny', icon: '☀️' },
        cloudy: { label: 'Cloudy', icon: '☁️' },
        rainy: { label: 'Rainy', icon: '🌧️' },
        storm: { label: 'Storm', icon: '⛈️' },
        fog: { label: 'Foggy', icon: '🌫️' }
    };

    const cond = condMap[data.condition] || (rainProb > 50 ? condMap.rainy : condMap.sunny);

    return {
        predicted_temperature: Math.round((tempSeries.reduce((a,b)=>a+b,0)/days) * 10) / 10,
        predicted_humidity: Math.round(humSeries.reduce((a,b)=>a+b,0)/days),
        rain_probability: rainProb,
        wind_speed_forecast: Math.round((windSeries.reduce((a,b)=>a+b,0)/days) * 10) / 10,
        condition_label: cond.label,
        condition_icon: cond.icon,
        confidence: Math.max(65, 95 - days * 2),
        series: {
            temperature: tempSeries,
            humidity: humSeries,
            windSpeed: windSeries
        }
    };
}

function displayResults(res, inputData) {
    const label = inputData.forecast_days === 1 ? 'Next Day' : `${inputData.forecast_days}-Day`;
    $('forecastBadge').textContent = label;
    $('chartBadge').textContent = `${label} · ${inputData.location}`;

    $('res-temp').textContent = res.predicted_temperature ?? '—';
    $('res-humidity').textContent = res.predicted_humidity ?? '—';
    $('res-rain').textContent = res.rain_probability ?? '—';
    $('res-wind').textContent = res.wind_speed_forecast ?? '—';
    $('res-condition').textContent = res.condition_label ?? '—';
    if (res.condition_icon) $('res-condition-icon').textContent = res.condition_icon;

    const conf = res.confidence || 90;
    $('confidenceBar').style.width = `${conf}%`;
    $('confidenceLabel').textContent = `${conf}% model confidence`;
}

function updateCharts(res, inputData) {
    const days = inputData.forecast_days;
    const series = res.series || {};
    initCharts(days, series.temperature || [28], series.humidity || [60], series.windSpeed || [18]);
}

function initCharts(days, tempData, humData, windData) {
    const labels = Array.from({ length: days }, (_, i) => days === 1 ? 'Day 1' : `Day ${i + 1}`);

    // Destroy existing instances
    if (tempChartInstance) tempChartInstance.destroy();
    if (humChartInstance) humChartInstance.destroy();
    if (windChartInstance) windChartInstance.destroy();

    const tempCtx = $('tempChart').getContext('2d');
    tempChartInstance = new Chart(tempCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Temperature (°C)',
                data: tempData,
                borderColor: '#0ea5e9',
                backgroundColor: 'rgba(14, 165, 233, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    const humCtx = $('humidityChart').getContext('2d');
    humChartInstance = new Chart(humCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Humidity (%)',
                data: humData,
                backgroundColor: '#38bdf8',
                borderRadius: 6
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    const windCtx = $('windChart').getContext('2d');
    windChartInstance = new Chart(windCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Wind Speed (km/h)',
                data: windData,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.3
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

function setLoading(on) {
    state.isLoading = on;
    const btn = $('predictBtn');
    if (!btn) return;
    btn.disabled = on;
}
