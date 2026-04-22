function renderLeaderboardTable(jsonPath, containerId) {
  fetch(jsonPath)
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var headers = data[0];
      var rows = data.slice(1);

      var html = '<table class="js-sort-table" id="results-' + containerId + '">';
      html += '<tr>';
      headers.forEach(function(h) {
        html += '<td class="js-sort-number"><strong>' + h + '</strong></td>';
      });
      html += '</tr>';

      rows.forEach(function(row) {
        html += '<tr>';
        row.forEach(function(cell, i) {
          if (i === 1) {
            html += '<td><b>' + cell + '</b></td>';
          } else {
            html += '<td>' + cell + '</td>';
          }
        });
        html += '</tr>';
      });

      html += '</table>';
      document.getElementById(containerId).innerHTML = html;
    });
}

document.addEventListener('DOMContentLoaded', function() {
  renderLeaderboardTable('static/data/td_reranking.json', 'table-td-reranking');
  renderLeaderboardTable('static/data/tq_reranking.json', 'table-tq-reranking');
  renderLeaderboardTable('static/data/td_retrieval.json', 'table-td-retrieval');
  renderLeaderboardTable('static/data/tq_retrieval.json', 'table-tq-retrieval');
});
