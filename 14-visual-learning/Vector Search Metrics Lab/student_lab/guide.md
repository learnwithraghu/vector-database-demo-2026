# Lab instructor guide — hosting on Amazon S3

This folder is a **static web lab** (`student_lab/index.html`). There is nothing for students to install.

## Host the student lab with a public URL

1. **Create an S3 bucket** (or use an existing one) in the AWS account your class uses.

2. **Upload** the contents of `student_lab/` — at minimum `index.html`. Keep the same object key if you link to `…/index.html` directly, or set the bucket’s **index document** to `index.html` if you use website hosting on the bucket root.

3. **Enable static website hosting** on the bucket (or serve the objects through **CloudFront** in front of S3 — both work).

4. **Allow public read access** to the objects students need:
   - Either a bucket policy that allows `s3:GetObject` for `Principal: "*"` on the relevant prefix (typical for a public static site), **or**
   - CloudFront with an origin access setup and a public distribution URL.

5. **Block public access**: turn off “Block *all* public access” only if you are intentionally making this lab world-readable; otherwise restrict by IP/VPN or use CloudFront with signed URLs if your institution requires it.

6. **Share the URL** with students — for example:
   - Website endpoint: `http://<bucket-name>.s3-website-<region>.amazonaws.com/`
   - Or your CloudFront domain: `https://dxxxx.cloudfront.net/`

Students open that URL in a normal browser; no backend is required.
