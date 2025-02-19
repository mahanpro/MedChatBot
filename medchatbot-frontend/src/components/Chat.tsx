'use client'
import React, { useState } from 'react';
import { Box, Button, Container, Paper, TextField, Typography, CircularProgress } from '@mui/material';
import axios from 'axios';

const Chat: React.FC = () => {
  const [input, setInput] = useState('');
  const [answer, setAnswer] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setAnswer(null); // Clear previous answer

    try {
      const response = await axios.post('http://localhost:8000/chat', { query: input });
      const { answer } = response.data;
      setAnswer(answer);
    } catch (error) {
      console.error('Error fetching chat response:', error);
      setAnswer('Sorry, there was an error processing your request.');
    } finally {
      setLoading(false);
      setInput(''); // Clear the input field
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        MedChatBot
      </Typography>
      <Paper elevation={3} sx={{ p: 2, minHeight: '30vh' }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '30vh' }}>
            <CircularProgress />
          </Box>
        ) : answer ? (
          <Typography variant="body1">{answer}</Typography>
        ) : (
          <Typography variant="body1" color="textSecondary">
            Your answer will appear here...
          </Typography>
        )}
      </Paper>
      <Box sx={{ display: 'flex', mt: 2 }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Type your question here..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              handleSend();
            }
          }}
        />
        <Button variant="contained" color="primary" onClick={handleSend} disabled={loading} sx={{ ml: 2 }}>
          Send
        </Button>
      </Box>
    </Container>
  );
};

export default Chat;
