
import { Client } from '@replit/object-storage';

const client = new Client();

export const uploadImage = async (imageFile: Buffer, filename: string): Promise<string> => {
  const key = `images/${Date.now()}-${filename}`;
  await client.put(key, imageFile);
  const url = await client.getSignedUrl(key);
  return url;
};

export const deleteImage = async (key: string): Promise<void> => {
  await client.delete(key);
};
