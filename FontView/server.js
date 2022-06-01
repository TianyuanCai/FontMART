const express = require("express");
const AWS = require("aws-sdk");

const app = express();
const path = require("path");
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "pug");

s3 = new AWS.S3({ apiVersion: "2006-03-01" });
const params = { Bucket: "font-metrics", Delimiter: "/", Prefix: "font-ttf/" };

app.use("/fonts", express.static(__dirname + "/fonts"));

app.get("/", (req, res) => {
  // get a list of font files in s3
  let allKeys = [];
  s3.listObjectsV2(params, function (err, data) {
    if (err) {
      console.log(err, err.stack); // an error occurred
    } else {
      let contents = data.Contents;
      contents.forEach(function (content) {
        allKeys.push(content.Key);
      });
    }

    // render template with s3 font keys
    res.render("index.pug", { font_list: JSON.stringify(allKeys) });
  });
});

// const PORT = 8080;
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`App is listening on Port ${PORT}!`);
});
