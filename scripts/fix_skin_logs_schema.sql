-- Fix skin_logs.image_path column to support base64 images
-- Run this SQL script to update your database

ALTER TABLE skin_logs MODIFY image_path TEXT NULL;
