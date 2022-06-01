let font_metrics_full = [];

function contourToString(contour) {
  return (
    '<pre class="contour">' +
    contour
      .map(function (point) {
        return (
          '<span class="' +
          (point.onCurve ? "on" : "off") +
          'curve">x=' +
          point.x +
          " y=" +
          point.y +
          "</span>"
        );
      })
      .join("\n") +
    "</pre>"
  );
}

function calculate_contrast(pathNode_outer, pathNode_inner) {
  let pathLength_outer = pathNode_outer.getTotalLength();
  let pathLength_inner = pathNode_inner.getTotalLength();

  let precision_outer = 8;
  let min_dist = Infinity,
    max_dist = 0;

  for (
    let scanLength_outer = 0;
    scanLength_outer <= pathLength_outer;
    scanLength_outer += precision_outer
  ) {
    let precision = 8;
    let best;
    let bestLength;
    let bestDistance = Infinity;
    let point_outer = pathNode_outer.getPointAtLength(scanLength_outer);

    function euclidean_distance(point1, point2) {
      let dx = point1.x - point2.x,
        dy = point1.y - point2.y;
      return Math.sqrt(dx * dx + dy * dy);
    }

    // linear scan for coarse approximation
    for (
      let scan, scanLength = 0, scanDistance;
      scanLength <= pathLength_inner;
      scanLength += precision
    ) {
      if (
        (scanDistance = euclidean_distance(
          (scan = pathNode_inner.getPointAtLength(scanLength)),
          point_outer
        )) < bestDistance
      ) {
        best = scan;
        bestLength = scanLength;
        bestDistance = scanDistance;
      }
    }

    precision /= 2;
    while (precision > 0.5) {
      let before,
        after,
        beforeLength,
        afterLength,
        beforeDistance,
        afterDistance;
      if (
        (beforeLength = bestLength - precision) >= 0 &&
        (beforeDistance = euclidean_distance(
          (before = pathNode_inner.getPointAtLength(beforeLength)),
          point_outer
        )) < bestDistance
      ) {
        (best = before),
          (bestLength = beforeLength),
          (bestDistance = beforeDistance);
      } else if (
        (afterLength = bestLength + precision) <= pathLength_inner &&
        (afterDistance = euclidean_distance(
          (after = pathNode_inner.getPointAtLength(afterLength)),
          point_outer
        )) < bestDistance
      ) {
        (best = after),
          (bestLength = afterLength),
          (bestDistance = afterDistance);
      } else {
        precision /= 2;
      }
    }

    best = [best.x, best.y];
    best.distance = bestDistance;

    max_dist = best.distance > max_dist ? best.distance : max_dist;
    min_dist = best.distance < min_dist ? best.distance : min_dist;
  }

  // scaled to the range 0 - 1, closer to 1, higher the contrast
  return [min_dist, max_dist];
}

function downloadCSV(csv, filename) {
  let csvFile;
  let downloadLink;

  csvFile = new Blob([csv], { type: "text/csv" });
  downloadLink = document.createElement("a");
  downloadLink.download = filename;
  downloadLink.href = window.URL.createObjectURL(csvFile);
  downloadLink.style.display = "none";
  document.body.appendChild(downloadLink);
  downloadLink.click();
}

function arrayToCSV(objArray) {
  const array = typeof objArray !== "object" ? JSON.parse(objArray) : objArray;
  let str =
    `${Object.keys(array[0])
      .map((value) => `"${value}"`)
      .join(",")}` + "\r\n";

  return array.reduce((str, next) => {
    str +=
      `${Object.values(next)
        .map((value) => `"${value}"`)
        .join(",")}` + "\r\n";
    return str;
  }, str);
}

function exportTableToCSV(filename) {
  // let json = font_metrics_full.items
  csv = arrayToCSV(font_metrics_full);
  downloadCSV(csv, filename);
}

function convertCanvasToImage(canvas) {
  let image = new Image();
  image.src = canvas.toDataURL("image/png");
  return image;
}

function getStandardDeviation(array) {
  const n = array.length;
  const mean = array.reduce((a, b) => a + b) / n;
  return Math.sqrt(
    array.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n
  );
}

function illustrate_font(font_file = "fonts/garamond.ttf") {
  opentype.load(font_file, function (err, font) {
    if (err) {
      alert("Font could not be loaded: " + err);
    } else {
      const font_name = font_file.split("/").slice(-1)[0].split(".")[0];

      let container = document.createElement("div");
      container.id = font_name;

      const font_size = 16; // size of the text in pixels
      const display_font_size = 72;
      const canvas = document.createElement("canvas");
      canvas.id = "canvas" + font_name;
      canvas.height = 100;
      canvas.width = 500;

      const ctx = canvas.getContext("2d");

      const test_string = "HandyGloves";
      const alphabets_lower = "abcdefghijklmnopqrstuvwxyz";
      const alphabets_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      const path = font.getPath(test_string, 0, 70, display_font_size);
      const glyph_alphabets_lower = font.getPath(
        alphabets_lower,
        0,
        70,
        font_size
      );
      path.draw(ctx);

      // create glyph
      const glyph_o = font.charToGlyph("o");
      const glyph_path = glyph_o.getPath(5, 200, font_size);

      // two different paths
      const path_data = glyph_path.toPathData(4);
      const inner_path = "M" + path_data.split("M")[1];
      const outer_path = "M" + path_data.split("M")[2];

      // the svg for getting path
      const path_outline_outer = document.querySelector("#outlinePath_outer");
      path_outline_outer.setAttribute("d", outer_path);
      const path_outline_inner = document.querySelector("#outlinePath_inner");
      path_outline_inner.setAttribute("d", inner_path);

      // function
      const min_max_distance = calculate_contrast(
        path_outline_outer,
        path_outline_inner
      );
      const contrast = 1 - min_max_distance[0] / min_max_distance[1];

      // additional font metrics
      lower_x_bounding_box = font
        .charToGlyph("x")
        .getPath(5, 300, font_size)
        .getBoundingBox();
      upper_x_bounding_box = font
        .charToGlyph("X")
        .getPath(5, 300, font_size)
        .getBoundingBox();
      const lower_x_height = lower_x_bounding_box.y2 - lower_x_bounding_box.y1;
      const lower_x_width = lower_x_bounding_box.x2 - lower_x_bounding_box.x1;
      const upper_x_height = upper_x_bounding_box.y2 - upper_x_bounding_box.y1;
      const upper_x_width = upper_x_bounding_box.x2 - upper_x_bounding_box.x1;

      const h_bouding_box = font
        .charToGlyph("h")
        .getPath(5, 300, font_size)
        .getBoundingBox();
      const g_bouding_box = font
        .charToGlyph("g")
        .getPath(5, 300, font_size)
        .getBoundingBox();
      const o_bounding_box = font
        .charToGlyph("o")
        .getPath(5, 300, font_size)
        .getBoundingBox();
      const o_width = o_bounding_box.x2 - o_bounding_box.x1;

      let ascender_letters = ["b", "d", "f", "h", "i", "k", "l", "t", "j"];
      let descender_letters = ["g", "q", "p", "y", "j"];
      let lower_width_array = [];
      let upper_width_array = [];
      let lower_height_array = [];
      let upper_height_array = [];
      let lower_ascender_array = [];
      let lower_descender_array = [];
      alphabets_lower.split("").forEach(function (element) {
        let tmp_bounding_box_lower = font
          .charToGlyph(element)
          .getPath(5, 300, font_size)
          .getBoundingBox();
        let tmp_bounding_box_upper = font
          .charToGlyph(element.toUpperCase())
          .getPath(5, 300, font_size)
          .getBoundingBox();

        let tmp_font_width_lower =
          tmp_bounding_box_lower.x2 - tmp_bounding_box_lower.x1;
        let tmp_font_width_upper =
          tmp_bounding_box_upper.x2 - tmp_bounding_box_upper.x1;

        let tmp_font_height_lower =
          tmp_bounding_box_lower.y2 - tmp_bounding_box_lower.y1;
        let tmp_font_height_upper =
          tmp_bounding_box_upper.y2 - tmp_bounding_box_upper.y1;

        lower_width_array.push(tmp_font_width_lower);
        upper_width_array.push(tmp_font_width_upper);

        if (ascender_letters.includes(element)) {
          lower_ascender_array.push(
            tmp_bounding_box_lower.y2 -
              tmp_bounding_box_lower.y1 -
              (lower_x_bounding_box.y2 - lower_x_bounding_box.y1)
          );
        } else if (descender_letters.includes(element)) {
          lower_descender_array.push(
            tmp_bounding_box_lower.y2 -
              tmp_bounding_box_lower.y1 -
              (lower_x_bounding_box.y2 - lower_x_bounding_box.y1)
          );
        } else {
          lower_height_array.push(tmp_font_height_lower);
          upper_height_array.push(tmp_font_height_upper);
        }
      });

      const reducer = (accumulator, currentValue) => accumulator + currentValue;

      let alphabets_lower_bb = glyph_alphabets_lower.getBoundingBox();

      const font_metrics = {
        "font-name": font_name,
        "stroke-contrast": contrast,
        weight: min_max_distance[1] / o_width,
        "avg-ascender-length":
          lower_ascender_array.reduce(reducer) / lower_ascender_array.length,
        "avg-descender-length":
          lower_descender_array.reduce(reducer) / lower_descender_array.length,
        "avg-char-height-lower":
          lower_height_array.reduce(reducer) / lower_height_array.length,
        "avg-char-width-lower":
          lower_width_array.reduce(reducer) / lower_width_array.length,
        "avg-char-height-relative":
          lower_height_array.reduce(reducer) /
          upper_height_array.reduce(reducer),
        "avg-char-width-relative":
          lower_width_array.reduce(reducer) / upper_width_array.reduce(reducer),
        "lower-width-sd": getStandardDeviation(lower_width_array),
        "lower-char-space":
          (alphabets_lower_bb.x2 -
            alphabets_lower_bb.x1 -
            lower_width_array.reduce(reducer)) /
          (lower_width_array.length - 1),
      };
      $.each(font_metrics, function (key, value) {
        font_metrics[key] =
          typeof value != "string" ? Math.round(value * 1e3) / 1e3 : value;
      });

      font_metrics_full.push(font_metrics);

      // append table headers
      if (!$("#font-metrics tbody").length) {
        $("#font-metrics").append("<thead><tr>");
        $("#font-metrics tr:last").append("<th>" + "canvas" + "</th>");
        $.each(font_metrics, function (key, value) {
          $("#font-metrics tr:last").append("<th>" + key + "</th>");
        });
        tbody = $("#font-metrics").append("<tbody/>");
      }

      // append font data
      trow = $("#font-metrics tbody:last").append("<tr><td>");
      $("#font-metrics td:last").append(convertCanvasToImage(canvas));
      $.each(font_metrics, function (key, value) {
        $("#font-metrics tr:last").append("<td>" + value + "</td>");
      });
    }
  });
}

$(document).ready(function () {
  let table = $("<table/>")
    .attr("id", "font-metrics")
    .addClass("table table-striped table-bordered")
    .attr("style", "width:100%")
    .attr("data-sorting", "true")
    .attr("data-filtering", "true");
  font_list.slice(1, 1).forEach(function (item, index) {
    illustrate_font(
      (font_file = "https://font-metrics.s3-us-west-1.amazonaws.com/" + item)
    );
  });

  // manual font specification
  illustrate_font((font_file = "fonts/open_sans.ttf"));

  $("#font-metrics-container").append(table);

  jQuery(function ($) {
    $(".table").footable();
  });
});
