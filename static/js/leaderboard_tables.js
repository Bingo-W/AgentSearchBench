function renderLeaderboardTable(jsonPath, containerId) {
  fetch(jsonPath)
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var headers = data[0];
      var rows = data.slice(1);

      var lastCol = headers.length - 1;

      // Find the per-column maximum for numeric metric columns (skip #, Model, Completeness)
      var maxPerCol = {};
      rows.forEach(function(row) {
        row.forEach(function(cell, i) {
          if (i === 0 || i === 1 || i === lastCol) return;
          var val = parseFloat(cell);
          if (!isNaN(val) && (maxPerCol[i] === undefined || val > maxPerCol[i])) {
            maxPerCol[i] = val;
          }
        });
      });

      var html = '<table class="js-sort-table" id="results-' + containerId + '">';
      html += '<tr>';
      headers.forEach(function(h) {
        html += '<td class="js-sort-number"><strong>' + h + '</strong></td>';
      });
      html += '</tr>';

      rows.forEach(function(row, rowIndex) {
        var isTop = rowIndex === 0;
        html += isTop ? '<tr class="top-row">' : '<tr>';
        row.forEach(function(cell, i) {
          var isBest = maxPerCol[i] !== undefined && parseFloat(cell) === maxPerCol[i];
          if (i === 1) {
            var label = isTop ? '🥇 ' + cell : cell;
            html += '<td><b>' + (isBest ? '<u>' + label + '</u>' : label) + '</b></td>';
          } else {
            html += '<td>' + (isBest ? '<u>' + cell + '</u>' : cell) + '</td>';
          }
        });
        html += '</tr>';
      });

      html += '</table>';
      document.getElementById(containerId).innerHTML = html;
      if (typeof sortTable !== 'undefined') sortTable.init();
    });
}

document.addEventListener('DOMContentLoaded', function() {
  renderLeaderboardTable('static/data/td_reranking.json', 'table-td-reranking');
  renderLeaderboardTable('static/data/tq_reranking.json', 'table-tq-reranking');
  renderLeaderboardTable('static/data/td_retrieval.json', 'table-td-retrieval');
  renderLeaderboardTable('static/data/tq_retrieval.json', 'table-tq-retrieval');
});
